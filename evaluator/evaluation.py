
import os

from dotenv import load_dotenv

from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams
)

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval
)

from deepeval.models import GeminiModel


load_dotenv()


class EvaluationEngine:

    def __init__(self):

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing from .env"
            )

        judge_name = os.getenv(
            "GEMINI_JUDGE_MODEL",
            "gemini-2.5-flash"
        )

        judge_model = GeminiModel(
            model=judge_name,
            api_key=api_key,
            temperature=0
        )

        self.metrics = [

            AnswerRelevancyMetric(
                threshold=0.7,
                model=judge_model
            ),

            FaithfulnessMetric(
                threshold=0.7,
                model=judge_model
            ),

            GEval(
                name="Answer Correctness",

                criteria=(
                    "Evaluate whether the actual answer is "
                    "factually correct and semantically consistent "
                    "with the expected answer. Accept different "
                    "wording when the meaning is correct."
                ),

                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT
                ],

                threshold=0.7,
                model=judge_model
            )
        ]

    def evaluate(
        self,
        question,
        actual_output,
        expected_output,
        context
    ):

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=context
        )

        results = []

        for metric in self.metrics:

            metric.measure(test_case)

            results.append({

                "metric": metric.__class__.__name__,

                "score": metric.score,

                "threshold": metric.threshold,

                "passed": metric.is_successful(),

                "reason": metric.reason

            })

        return results
