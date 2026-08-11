from __future__ import annotations

import os
import httpx


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.6-luna"


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY environment variable not set")
        self._headers = {
              "Authorization": f"Bearer {api_key}",
              "Content-Type": "application/json",
          }

    def complete(self, prompt: str) -> str:
          try:
              response = httpx.post(
                  OPENROUTER_API_URL,
                  headers=self._headers,
                  json={
                      "model": MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 200,
                  },
                  timeout=30.0,
              )
              response.raise_for_status()
              content = response.json()["choices"][0]["message"]["content"]
              if not content or not content.strip():
                  raise LLMError("Empty response from LLM")
              return content.strip()
          except httpx.TimeoutException as e:
              raise LLMError("LLM request timed out") from e
          except httpx.HTTPStatusError as e:
              raise LLMError(f"LLM request failed: {e.response.status_code}") from e
          except (KeyError, IndexError) as e:
              raise LLMError(f"Unexpected LLM response format: {e}") from e

    def rewrite_query(self, query: str) -> str:
        prompt = f"""You are helping search for recipes. Rewrite the following request into a short list of descriptive keywords that would match relevant recipes.

        Focus on: meal type, dietary style, cooking method, ingredients, time constraints, nutrition profile.
        Output only the keywords, no explanation, no bullet points, one line.

        Request: {query}
        Keywords:"""
        return self.complete(prompt)

    def explain_results(self, query: str, recipes: list[dict]) -> list[str]:
        recipe_list = "\n".join(
            f"{i+1}. {r['name']}: {r['text']}" for i, r in enumerate(recipes)
        )
        prompt = f"""A user searched for: "{query}"
        These recipes were recommended:
        {recipe_list}

        For each recipe, write exactly one short sentence explaining why it fits the request.
        Output only the sentences, one per line, in the same order.

        Explanations:
        """
        response = self.complete(prompt)
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return lines
