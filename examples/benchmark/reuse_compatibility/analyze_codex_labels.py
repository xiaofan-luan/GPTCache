"""Compare a full Codex relabel with the published labels and JEV baseline."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-labels", type=Path, default=ROOT / "codex_labels.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "codex_label_summary.json")
    parser.add_argument("--conflicts", type=Path, default=ROOT / "codex_label_conflicts.jsonl")
    args = parser.parse_args()

    cases = {row["uid"]: row for row in rows(ROOT / "cases.jsonl")}
    published = {row["uid"]: row for row in rows(ROOT / "labels.jsonl")}
    baseline = {row["uid"]: row for row in rows(ROOT / "baseline_jev_070.jsonl")}
    codex = {row["uid"]: row for row in rows(args.codex_labels)}
    assert len(codex) == len(cases), "Codex labels must cover all benchmark cases"
    assert set(codex) == set(cases) == set(published) == set(baseline)

    cross = Counter((published[uid]["label"], row["label"]) for uid, row in codex.items())
    reasons = Counter(row["reason_code"] for row in codex.values())
    labels = Counter(row["label"] for row in codex.values())
    confidences = Counter(row["confidence"] for row in codex.values())
    by_arm = defaultdict(Counter)
    by_split = defaultdict(Counter)
    outcomes = Counter()
    conflict_rows = []

    for uid, truth in codex.items():
        case = cases[uid]
        by_arm[case["arm_title"]][truth["label"]] += 1
        by_split[case["split"]][truth["label"]] += 1
        hit = baseline[uid]["decision"] == "reuse"
        if truth["label"] == "uncertain":
            outcome = "uncertain_reuse" if hit else "uncertain_refusal"
        elif hit:
            outcome = "correct_reuse" if truth["label"] == "reuse" else "wrong_reuse"
        else:
            outcome = "missed_reuse" if truth["label"] == "reuse" else "correct_refusal"
        outcomes[outcome] += 1
        if published[uid]["label"] != truth["label"]:
            conflict_rows.append({
                **case,
                "published_label": published[uid]["label"],
                "published_reason_zh": published[uid]["reason_zh"],
                "codex_label": truth["label"],
                "codex_reason_code": truth["reason_code"],
                "codex_reason_zh": truth["reason_zh"],
                "codex_gates": {key: truth[key] for key in (
                    "task_compatible", "constraints_compatible", "format_complete",
                    "context_sufficient", "freshness_safe")},
                "codex_evidence_field": truth["evidence_field"],
                "codex_evidence_quote": truth["evidence_quote"],
                "codex_confidence": truth["confidence"],
                "jev_decision": baseline[uid]["decision"],
                "jev_score": baseline[uid]["score"],
            })

    correct = outcomes["correct_reuse"]
    wrong = outcomes["wrong_reuse"]
    missed = outcomes["missed_reuse"]
    hit_count = correct + wrong + outcomes["uncertain_reuse"]
    summary = {
        "records": len(cases),
        "codex_model": next(iter(codex.values()))["model"],
        "codex_reasoning_effort": next(iter(codex.values()))["reasoning_effort"],
        "codex_prompt_sha256": next(iter(codex.values()))["prompt_sha256"],
        "codex_labels": dict(labels),
        "codex_confidences": dict(confidences),
        "codex_reason_codes": dict(reasons),
        "published_vs_codex": {
            f"{old}_to_{new}": count for (old, new), count in sorted(cross.items())
        },
        "exact_label_agreement": sum(
            published[uid]["label"] == truth["label"] for uid, truth in codex.items()),
        "conflicts": len(conflict_rows),
        "by_arm": {name: dict(counts) for name, counts in by_arm.items()},
        "by_split": {name: dict(counts) for name, counts in by_split.items()},
        "jev_070_against_codex": {
            "hits": hit_count,
            "hit_rate": ratio(hit_count, len(cases)),
            **dict(outcomes),
            "reuse_precision": ratio(correct, correct + wrong),
            "reuse_recall": ratio(correct, correct + missed),
        },
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with args.conflicts.open("w") as stream:
        for row in conflict_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
