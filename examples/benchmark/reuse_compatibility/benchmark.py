"""Run and score the GPTCache reuse-compatibility benchmark."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    manifest = json.loads((ROOT/"manifest.json").read_text())
    cases = rows(ROOT/"cases.jsonl"); labels = rows(ROOT/"labels.jsonl")
    assert len(cases) == len(labels) == manifest["records"] == 1536
    assert len({r["uid"] for r in cases}) == len(cases)
    assert {r["uid"] for r in cases} == {r["uid"] for r in labels}
    assert Counter(r["label"] for r in labels) == Counter(manifest["label_counts"])
    prompt_source = ROOT.parents[2]/manifest["jev_prompt_source"]
    prompt_source_available = prompt_source.exists()
    if prompt_source_available:
        assert sha256(prompt_source) == manifest["jev_prompt_source_sha256"]
    for name, info in manifest["files"].items():
        path = ROOT/name
        assert path.exists() and sha256(path) == info["sha256"], name
        if "rows" in info: assert len(rows(path)) == info["rows"], name
    print(json.dumps({"valid": True, "records": len(cases),
                      "prompt_source_available": prompt_source_available,
                      "labels": dict(Counter(r["label"] for r in labels))}, ensure_ascii=False))


def score(predictions, threshold, labels_path=None):
    labels_path = labels_path or ROOT/"labels.jsonl"
    labels = {r["uid"]: r for r in rows(labels_path)}
    predicted = {r["uid"]: r for r in rows(predictions)}
    assert set(predicted) == set(labels), "predictions must cover every benchmark UID exactly"
    counts = Counter(); errors = 0
    for uid, truth in labels.items():
        row = predicted[uid]; errors += bool(row.get("error"))
        if "score" in row:
            hit = not row.get("error", False) and float(row["score"]) >= threshold
        else:
            hit = row["decision"] == "reuse"
        label = truth["label"]
        if label == "uncertain": outcome = "uncertain_reuse" if hit else "uncertain_refusal"
        elif hit: outcome = "correct_reuse" if label == "reuse" else "wrong_reuse"
        else: outcome = "missed_reuse" if label == "reuse" else "correct_refusal"
        counts[outcome] += 1
    hits = counts["correct_reuse"] + counts["wrong_reuse"] + counts["uncertain_reuse"]
    result = {
        "records": len(labels), "threshold": threshold, "api_errors": errors,
        "hits": hits, "hit_rate": hits/len(labels), **dict(counts),
        "reuse_precision": counts["correct_reuse"]/(counts["correct_reuse"]+counts["wrong_reuse"]),
        "reuse_recall": counts["correct_reuse"]/(counts["correct_reuse"]+counts["missed_reuse"]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def evaluate(case):
    sys.path.insert(0, str(ROOT.parents[2]))
    from gptcache.similarity_evaluation.jev import JevEvaluation

    evaluator = JevEvaluation(timeout=45)
    started = time.perf_counter()
    score_value = evaluator.evaluation(
        {"question": case["current_request"]},
        {"question": case["cached_request"], "answer": case["cached_answer"]})
    scores = evaluator.last_scores
    valid = set(scores) == set(evaluator.DIMENSIONS) and all(0 <= v <= 1 for v in scores.values())
    return {"uid": case["uid"], "score": score_value, "scores": scores,
            "error": not valid, "seconds": time.perf_counter()-started}


def run(output, threshold, workers):
    cases = rows(ROOT/"cases.jsonl")
    prior = rows(output) if output.exists() else []
    done = {row["uid"]: row for row in prior if not row.get("error")}
    todo = [case for case in cases if case["uid"] not in done]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, case) for case in todo]
        for future in as_completed(futures):
            result = future.result(); stream.write(json.dumps(result)+"\n"); stream.flush()
            done[result["uid"]] = result
            if len(done) % 64 == 0: print("evaluated", len(done), "/", len(cases), flush=True)
    assert len(done) == len(cases)
    score(output, threshold)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    scoring = sub.add_parser("score"); scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--labels", type=Path)
    scoring.add_argument("--threshold", type=float, default=0.70)
    running = sub.add_parser("run"); running.add_argument("--output", type=Path, required=True)
    running.add_argument("--threshold", type=float, default=0.70)
    running.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "validate": validate()
    elif args.command == "score": score(args.predictions, args.threshold, args.labels)
    else: run(args.output, args.threshold, args.workers)


if __name__ == "__main__": main()
