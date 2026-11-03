
import json

import pytest

from pathlib import Path

from evaluator.llm_client import LLMClient
from evaluator.evaluation import EvaluationEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "datasets" / "golden_dataset.json"


with open(DATASET_PATH, "r", encoding="utf-8") as file:
    DATASET = json.load(file)


@pytest.fixture(scope="module")
def llm():

    return LLMClient()


@pytest.fixture(scope="module")
def evaluator():

    return EvaluationEngine()


@pytest.mark.parametrize(
    "test",
    DATASET,
    ids=lambda test: test["id"]
)
def test_llm_response_quality(test, llm, evaluator):

    answer = llm.generate_response(
        test["input"],
        test["context"]
    )

    assert answer, "Gemini returned an empty response"

    results = evaluator.evaluate(
        question=test["input"],
        actual_output=answer,
        expected_output=test["expected_output"],
        context=test["context"]
    )

    for metric in results:

        assert metric["passed"], (
            f"{metric['metric']} failed. "
            f"Score: {metric['score']}. "
            f"Reason: {metric['reason']}"
        )
