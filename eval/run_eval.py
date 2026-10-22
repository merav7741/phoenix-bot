"""Run the evaluation questions against the bot and report pass/fail.

Usage: python eval/run_eval.py
Writes every answer to eval/results.json for manual review.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.rag import RAGError, get_answer  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


def load_cases():
    cases = []
    for item in json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8")):
        expect = "answer" if item["type"] == "answerable" else "refuse"
        cases.append({**item, "expect": expect})
    for item in json.loads(
        (EVAL_DIR / "negative_questions.json").read_text(encoding="utf-8")
    ):
        cases.append({**item, "type": "negative"})
    return cases


def run_case(case):
    try:
        result = get_answer(case["q"])
    except RAGError as exc:
        return {**case, "error": str(exc), "passed": False}
    sources = [source["url"] for source in result.get("sources", [])]
    answered = bool(sources)
    passed = answered == (case["expect"] == "answer")
    expected_part = case.get("expected_url_part")
    if passed and answered and expected_part:
        passed = any(expected_part in url for url in sources)
    return {
        **case,
        "answer": result.get("answer") or result.get("clarification_question"),
        "sources": sources,
        "passed": passed,
    }


def main():
    cases = load_cases()
    results = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(run_case, case) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"{len(results)}/{len(cases)} {mark} [{result['type']}] {result['q']}")
            if not result["passed"]:
                print(f"     sources: {result.get('sources')} error: {result.get('error')}")
            (EVAL_DIR / "results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    passed = sum(result["passed"] for result in results)
    print(f"\n{passed}/{len(results)} passed")


if __name__ == "__main__":
    main()
