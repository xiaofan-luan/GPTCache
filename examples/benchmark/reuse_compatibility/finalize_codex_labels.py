"""Combine Codex first-pass agreements with high-effort conflict adjudication."""
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
    parser.add_argument("--output", type=Path,
                        default=ROOT / "codex_adjudicated_labels.jsonl")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "codex_adjudicated_summary.json")
    args = parser.parse_args()

    cases_list = rows(ROOT / "cases.jsonl")
    cases = {row["uid"]: row for row in cases_list}
    published = {row["uid"]: row for row in rows(ROOT / "labels.jsonl")}
    first = {row["uid"]: row for row in rows(ROOT / "codex_labels.jsonl")}
    adjudicated = {
        row["uid"]: row for row in rows(ROOT / "codex_conflict_adjudication.jsonl")
    }
    baseline = {row["uid"]: row for row in rows(ROOT / "baseline_jev_070.jsonl")}
    assert set(cases) == set(published) == set(first) == set(baseline)
    conflicts = {
        uid for uid in cases if published[uid]["label"] != first[uid]["label"]
    }
    assert set(adjudicated) == conflicts

    final = []
    triad = Counter()
    for case in cases_list:
        uid = case["uid"]
        if uid in conflicts:
            row = adjudicated[uid]
            source = "codex_high_conflict_adjudication"
            triad[(published[uid]["label"], first[uid]["label"], row["label"])] += 1
        else:
            row = first[uid]
            source = "published_codex_first_pass_agreement"
        final.append({**row, "label_source": source})

    with args.output.open("w") as stream:
        for row in final:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    labels = Counter(row["label"] for row in final)
    reasons = Counter(row["reason_code"] for row in final)
    confidences = Counter(row["confidence"] for row in final)
    sources = Counter(row["label_source"] for row in final)
    final_by_uid = {row["uid"]: row for row in final}
    by_arm = defaultdict(Counter)
    outcomes = Counter()
    for uid, row in final_by_uid.items():
        by_arm[cases[uid]["arm_title"]][row["label"]] += 1
        hit = baseline[uid]["decision"] == "reuse"
        if row["label"] == "uncertain":
            outcome = "uncertain_reuse" if hit else "uncertain_refusal"
        elif hit:
            outcome = "correct_reuse" if row["label"] == "reuse" else "wrong_reuse"
        else:
            outcome = "missed_reuse" if row["label"] == "reuse" else "correct_refusal"
        outcomes[outcome] += 1
    correct = outcomes["correct_reuse"]
    wrong = outcomes["wrong_reuse"]
    missed = outcomes["missed_reuse"]
    hits = correct + wrong + outcomes["uncertain_reuse"]
    summary = {
        "records": len(final),
        "protocol": {
            "agreement_rows": len(final) - len(conflicts),
            "conflict_rows_adjudicated": len(conflicts),
            "first_pass_model": next(iter(first.values()))["model"],
            "first_pass_effort": next(iter(first.values()))["reasoning_effort"],
            "adjudication_model": next(iter(adjudicated.values()))["model"],
            "adjudication_effort": next(iter(adjudicated.values()))["reasoning_effort"],
            "prompt_sha256": next(iter(first.values()))["prompt_sha256"],
        },
        "labels": dict(labels),
        "reason_codes": dict(reasons),
        "confidences": dict(confidences),
        "label_sources": dict(sources),
        "conflict_triad": {
            f"published_{old}__first_{initial}__adjudicated_{final_label}": count
            for (old, initial, final_label), count in sorted(triad.items())
        },
        "final_vs_published": dict(Counter(
            f"published_{published[uid]['label']}__final_{row['label']}"
            for uid, row in final_by_uid.items()
        )),
        "by_arm": {name: dict(counts) for name, counts in by_arm.items()},
        "jev_070_against_codex_adjudicated": {
            "hits": hits,
            "hit_rate": ratio(hits, len(final)),
            **dict(outcomes),
            "reuse_precision": ratio(correct, correct + wrong),
            "reuse_recall": ratio(correct, correct + missed),
        },
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
