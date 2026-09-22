# GPTCache Reuse Compatibility Benchmark

这个 benchmark 判断一个缓存回答能否**原样复用于当前请求**。它只检查：

1. 当前请求与缓存请求的任务、实体、数字、版本和显式约束是否兼容；
2. 缓存回答是否满足语言、格式、长度、数量、语气和结构完整性要求；
3. 是否依赖缺失的历史对话、附件、位置或私有信息；
4. 是否存在价格、汇率、日程、可用性、现行规则等时效风险。

它**不判断**答案中的事实、计算、代码、推理、医疗或法律内容是否正确。两个请求相同但旧答案本身答错时，在这个 benchmark 中仍可标为 `reuse`。这使它专门衡量 Cache 路由策略，而不是回答质量。

## 依赖关系

本PR只包含benchmark数据、标注、基线结果和评测工具，对应的JEV实现位于PR [#701](https://github.com/zilliztech/GPTCache/pull/701)。`validate`和`score`可在本PR上独立运行；重新调用JEV生成分数的`run`命令需要先合并或检出#701。

## 数据规模

共1,536条当前请求/缓存请求/缓存回答三元组：

| 数据组 | 校准集 | 测试集 | 合计 |
| --- | ---: | ---: | ---: |
| LmArena / GPT-4.1 nano回答 | 128 | 384 | 512 |
| SearchQueries / Llama-3-8B回答 | 128 | 384 | 512 |
| SearchQueries / GPT-4o mini回答 | 128 | 384 | 512 |
| 合计 | 384 | 1,152 | 1,536 |

输入来自固定版本的 vCache 数据集：

- `vCache/SemBenchmarkLmArena`：`c6f36229bef6a5f7936e5b9d982bd5766ef2f862`
- `vCache/SemBenchmarkSearchQueries`：`5159bff8ccf5d7c57dd82cbb24a76e2e5d24f1c0`

样本沿用原实验已经冻结的候选配对，没有重新检索、重新生成回答或更改 calibration/test 划分。

## 标注方法

- 全部1,536条由独立于JEV的模型在看不到JEV分数和原始等价类标签的情况下逐条标注；
- 首轮与GPTCache决定冲突的236条，由第二个模型重新盲审；
- 6条把“答案正确性”误当成“缓存兼容性”的边界案例经过人工范围修正；
- 最终标签：`reuse` 505条、`reject` 1,027条、`uncertain` 4条。

这些标签是可复现的模型辅助标注，并非完整人工事实金标。原因和原文证据随每条标签保存，便于继续复核。

## 文件

- `cases.jsonl`：1,536条输入；包含当前请求、缓存请求和缓存回答。
- `labels.jsonl`：最终兼容性标签、中文原因、原因类别、证据和标注来源。
- `baseline_jev_070.jsonl`：当前完整性与上下文增强版JEV prompt在0.70阈值下的五项分数与决定。
- `baseline_jev_075.jsonl`：上一个prompt版本在0.75阈值下的历史基线，保留用于回归比较。
- `codex_adjudicated_labels.jsonl`：Codex全量首标并对冲突样本复审后的候选标签；不替代`labels.jsonl`。
- `CODEX_LABELING.md`：Codex标注目标、两阶段协议、标签分布与限制。
- `codex_label.py`、`codex_label_schema.json`：结构化Codex批量标注器及输出Schema。
- `codex_labels.jsonl`、`codex_label_conflicts.jsonl`、`codex_conflict_adjudication.jsonl`：首轮标签、冲突上下文和high-reasoning复审结果。
- `analyze_codex_labels.py`、`finalize_codex_labels.py`：冲突分析、合并候选标签并重算指标。
- `summary.json`：标签分布和基线指标。
- `manifest.json`：来源版本、文件哈希、行数和评测协议。
- `benchmark.py`：校验数据、运行JEV或评分预测。
- `WRONG_REUSE_075.html`：上一个prompt与0.75阈值的历史误复用审计。
- `PROMPT_COMPARISON.md`：原prompt、宽松版、平衡版及不同阈值的指标对比。

## 当前基线

严格排除答案正确性后，当前增强版prompt与0.70阈值对已发布标签的结果：

| 指标 | 结果 |
| --- | ---: |
| 系统命中 | 366 / 1,536（23.8%） |
| 正确复用 | 364 |
| 误复用 | 1 |
| 漏复用 | 141 |
| 复用精确率 | 99.7% |
| 应复用召回率 | 72.1% |

另有1条系统命中和3条系统拒绝对应`uncertain`标签，不计入精确率和召回率。对Codex复审候选标签，同一结果为353条正确复用、10条误复用、98条漏复用，精确率97.3%，召回率78.3%。

## 使用

验证文件、哈希、标签和基线完整性：

```bash
.venv/bin/python examples/benchmark/reuse_compatibility/benchmark.py validate
```

重新计算随包提供的基线指标：

```bash
.venv/bin/python examples/benchmark/reuse_compatibility/benchmark.py score \
  --predictions examples/benchmark/reuse_compatibility/baseline_jev_070.jsonl
```

使用Codex复审候选标签评分：

```bash
.venv/bin/python examples/benchmark/reuse_compatibility/benchmark.py score \
  --predictions examples/benchmark/reuse_compatibility/baseline_jev_070.jsonl \
  --labels examples/benchmark/reuse_compatibility/codex_adjudicated_labels.jsonl
```

调用当前仓库的 `JevEvaluation` 重新评测全部1,536条：

```bash
JEV=... .venv/bin/python examples/benchmark/reuse_compatibility/benchmark.py run \
  --output /tmp/jev_predictions.jsonl --threshold 0.70 --workers 12
```

JEV调用逐条落盘，可使用同一个输出文件断点续跑。评分时要求预测文件完整覆盖1,536个UID；API错误按拒绝处理并单独计数。
基线使用的完整问题定义保存在本次提交的 `gptcache/similarity_evaluation/jev.py`，其SHA-256同时记录在 `manifest.json`。

## 指标定义

- `hit_rate`：系统决定复用的记录数 / 全部记录数；
- `reuse_precision`：正确复用 /（正确复用 + 误复用）；
- `reuse_recall`：正确复用 /（正确复用 + 漏复用）；
- `wrong_reuse`：系统复用，但兼容性标签为 `reject`；
- `missed_reuse`：系统拒绝，但兼容性标签为 `reuse`。

`uncertain` 不参与精确率和召回率分母，但会按系统放行或拒绝单独报告。

## 限制

- 固定样本来自语义检索后的候选配对，不代表线上请求分布或端到端命中率；
- SearchQueries随附回答存在大量截断、重复和无实质内容文本；
- 远端 `jev-latest` 是可变别名，未来重跑的分数可能变化；
- 标注理由不验证答案内容的事实正确性；
- 阈值0.70来自本数据集上的重新校准；应用到其他业务域时应使用独立校准集确认。
