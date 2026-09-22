# Codex 全量兼容性标注

## 标注目标

标注的唯一问题是：

> 在不修改、不补充、不重新验证缓存回答的前提下，能否把它原样作为当前请求的最终回复？

只判断五个 gate：

1. `task_compatible`：当前请求和缓存请求要求同一项实质任务和交付物；
2. `constraints_compatible`：实体、数字、日期、版本、地点、数量、否定和显式约束兼容；
3. `format_complete`：语言、格式、条目数、语气和结构满足显式要求，且回答没有明显首尾截断或空占位；
4. `context_sufficient`：不依赖缺失的历史消息、附件、私有数据、位置或无法衔接的续写前缀；
5. `freshness_safe`：不需要未获支持的实时价格、天气、日程、库存、可用性、现任人员或当前产品线。

五项全为 `yes` 才标 `reuse`；任一项明确为 `no` 标 `reject`；没有 `no`、但至少一项无法确定时标 `uncertain`。

## 明确排除

不判断缓存回答中的事实、计算、代码、推理、建议、例子或结论是否正确。同一问题得到一个事实错误的回答，只要五项兼容 gate 均满足，仍标 `reuse`。完整的拒绝回复也可以复用；拒绝策略是否正确属于回答质量，不属于缓存兼容性。

结构完整性只检查交付物是否存在，不检查它是否正确。例如，回答给出一个人名但该人实际上不符合条件，属于内容正确性，仍可复用；回答只说“答案很多且很准确”却没有给出任何实际答案，则属于交付物缺失，应拒绝。

## Codex 标注协议

- 输入：固定的 1,536 条 `current_request`、`cached_request`、`cached_answer` 三元组；
- 首轮：`gpt-5.6-sol`，`medium` reasoning，对全部样本盲标；
- 输出：JSON Schema 强约束的标签、五项 gate、原因码、中文理由、原文证据和置信度；
- 校验：UID 不缺不重；`reuse` 必须五项全 `yes`；`reject` 必须至少一项 `no`；证据必须逐字存在于指定原字段；
- 冲突复审：首轮与已发布标签不同的 147 条，用同一 prompt、`high` reasoning 再标一次；
- 合并：1,389 条一致样本使用首轮结果，147 条冲突样本使用 high-reasoning 复审结果。

最终 prompt SHA-256：

```text
74f99fefe69593181ae149a7866b8d94d3b2ca17cdc69dd5fa255e0e126cb69c
```

## Codex 候选标签结果

| 标签 | 数量 |
| --- | ---: |
| `reuse` | 451 |
| `reject` | 1,079 |
| `uncertain` | 6 |
| 合计 | 1,536 |

主要拒绝原因：

| 原因 | 数量 |
| --- | ---: |
| 输出不完整 | 432 |
| 任务不匹配 | 300 |
| 请求约束不匹配 | 217 |
| 格式不匹配 | 75 |
| 缺失上下文 | 32 |
| 时效风险 | 29 |

按数据组划分：

| 数据组 | 复用 | 拒绝 | 不确定 |
| --- | ---: | ---: | ---: |
| LmArena / GPT-4.1 nano | 405 | 101 | 6 |
| SearchQueries / Llama-3-8B | 13 | 499 | 0 |
| SearchQueries / GPT-4o mini | 33 | 479 | 0 |

## 新版 JEV 0.70 对 Codex 候选标签

| 指标 | 结果 |
| --- | ---: |
| 系统命中 | 366 / 1,536（23.8%） |
| 正确复用 | 353 |
| 错误复用 | 10 |
| 漏复用 | 98 |
| 正确拒绝 | 1,069 |
| 不确定样本中的命中 / 拒绝 | 3 / 3 |
| 复用精确率 | 97.3% |
| 应复用召回率 | 78.3% |

## 文件

- `codex_label.py`：Codex CLI 批量标注器，可断点续跑；
- `codex_label_schema.json`：结构化输出约束；
- `codex_labels.jsonl`：全部 1,536 条 medium-reasoning 首轮标签；
- `codex_label_conflicts.jsonl`：首轮与已发布标签不同的 147 条及完整上下文；
- `codex_conflict_adjudication.jsonl`：147 条 high-reasoning 复审标签；
- `codex_adjudicated_labels.jsonl`：合并后的 Codex 候选标签；
- `codex_adjudicated_summary.json`：候选标签和 JEV 指标汇总；
- `analyze_codex_labels.py`、`finalize_codex_labels.py`：对比与合并脚本。

## 运行

全量首标：

```bash
.venv/bin/python examples/benchmark/reuse_compatibility/codex_label.py \
  --output examples/benchmark/reuse_compatibility/codex_labels.jsonl \
  --batch-size 24 --workers 4 --effort medium
```

冲突复审：

```bash
.venv/bin/python examples/benchmark/reuse_compatibility/codex_label.py \
  --output examples/benchmark/reuse_compatibility/codex_conflict_adjudication.jsonl \
  --uids-from examples/benchmark/reuse_compatibility/codex_label_conflicts.jsonl \
  --batch-size 12 --workers 4 --effort high
```

合并并重算指标：

```bash
.venv/bin/python examples/benchmark/reuse_compatibility/finalize_codex_labels.py
```

## 使用限制

这仍是模型辅助候选标签，不是完整人工金标。合并结果中有 127 条与已发布标签不同，主要争议集中在短回答是否已经构成完整交付物、长答案尾部是否属于实质截断，以及请求措辞差异是否构成硬约束。Codex 对部分“只输出第一项或第一步”的 SearchQueries 样本仍可能判断过宽，因此不应在没有冲突复核的情况下直接覆盖 `labels.jsonl`。
