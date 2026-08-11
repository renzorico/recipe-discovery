import pytest
from llm import LLMClient, LLMError
import os
import search as search_module
import json
from search import make_recipe_text

def test_llm_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        LLMClient()

def test_search_rejects_empty_query():
    with pytest.raises(ValueError, match="non-empty"):
        search_module.search("")

def test_search_rejects_invalid_k():
    with pytest.raises(ValueError):
        search_module.search("chicken soup", k=0)

def test_search_rejects_k_too_large():
    with pytest.raises(ValueError):
        search_module.search("chicken soup", k=101)

def test_search_returns_correct_shape():
    search_module.warmup()
    results = search_module.search("quick chicken dinner", k=5)

    assert len(results) <= 5
    for result in results:
        assert "recipe_id" in result
        assert "score" in result
        assert "explanation" in result
        assert isinstance(result["recipe_id"], str)
        assert isinstance(result["score"], float)
        assert isinstance(result["explanation"], str)


def test_make_recipe_text_returns_expected_fields():
    row = {
        "name": "Test Recipe",
        "RecipeCategory": "Chicken",
        "tags": json.dumps(["Quick", "Easy"]),
        "ingredients": json.dumps([{"name": "chicken"}, {"name": "lemon"}]),
        "cooking_time": 20,
        "preparation_time": 10,
        "calories": 250,
        "protein": 30,
    }
    text = make_recipe_text(row)

    assert "Test Recipe" in text
    assert "Chicken" in text
    assert "Quick" in text
    assert "chicken" in text
    assert "lemon" in text
    assert "30 minutes" in text
    assert "low-calorie" in text
    assert "high-protein" in text
