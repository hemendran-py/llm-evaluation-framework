
import json
import os

from datetime import datetime, timezone

from evaluator.llm_client import LLMClient
from evaluator.evaluation import EvaluationEngine


DATASET_PATH = "datasets/golden_dataset.json"

REPORT_DIR = "reports"

# Start with one run to check API quotas.
RUNS_PER_QUESTION = 1


def load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_report(results):

    os.makedirs(REPORT_DIR, exist_ok=True)

    filename = datetime.now(timezone.utc).strftime(
        "evaluation_%Y%m%d_%H%M%S.json"
    )

    path = os.path.join(REPORT_DIR, filename)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nReport saved to: {path}")


def main():

    print("\nInitializing Gemini LLM Evaluation Framework...")

    llm = LLMClient()

    evaluator = EvaluationEngine()

    dataset = load_dataset()

    all_results = []

    for test in dataset:

        for run in range(1, RUNS_PER_QUESTION + 1):

            print("\n" + "=" * 50)

            print(f"Test ID: {test['id']}")
            print(f"Run: {run}")

            answer = llm.generate_response(
                test["input"],
                test["context"]
            )

            print("\nGenerated Answer:")
            print(answer)

            scores = evaluator.evaluate(
                question=test["input"],
                actual_output=answer,
                expected_output=test["expected_output"],
                context=test["context"]
            )

            passed = all(
                metric["passed"] for metric in scores
            )

            result = {

                "test_id": test["id"],

                "run": run,

                "question": test["input"],

                "actual_output": answer,

                "expected_output": test["expected_output"],

                "metrics": scores,

                "overall_passed": passed

            }

            all_results.append(result)

            print("\nEvaluation Results:")

            for metric in scores:

                print(
                    f"{metric['metric']}: "
                    f"{metric['score']} "
                    f"| Passed: {metric['passed']}"
                )

            print(
                "\nOverall:",
                "PASSED" if passed else "FAILED"
            )

    save_report(all_results)

    passed_count = sum(
        1 for result in all_results
        if result["overall_passed"]
    )

    failed_count = len(all_results) - passed_count

    print("\n" + "=" * 50)

    print("FINAL EVALUATION SUMMARY")

    print("=" * 50)

    print(f"Total Evaluations: {len(all_results)}")

    print(f"Passed: {passed_count}")

    print(f"Failed: {failed_count}")

    print(
        f"Success Rate: "
        f"{(passed_count / len(all_results) * 100):.2f}%"
    )


if __name__ == "__main__":
    main()
