
import os
import time

from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.errors import ServerError


# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)


class LLMClient:

    def __init__(self):

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing from .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )

    def generate_response(self, question, context):

        context_text = "\n".join(context)

        prompt = f"""
Use only the provided context to answer the question.

If the answer is unavailable in the context,
say that you do not have enough information.

Context:
{context_text}

Question:
{question}
"""

        # Retry up to 3 times for temporary server errors
        for attempt in range(3):

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=300
                    )
                )

                answer = response.text

                if not answer:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return answer.strip()

            except ServerError:

                if attempt == 2:
                    raise

                wait_time = 5 * (attempt + 1)

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Gemini failed after 3 attempts."
        )
