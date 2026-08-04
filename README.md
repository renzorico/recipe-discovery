# Recipe Discovery

You're building a recipe recommendation system with hundreds of thousands of recipes. Today it works by keyword: if the user does not hit the exact term, they find nothing.

We want them to ask for what they want the way they would say it out loud:

> *"something vegetarian and quick that doesn't need an oven"*

> *"I have chicken, a lemon and half an hour"*

> *"something comforting for a winter Sunday"*

**Your task:** build a recipe recommendation engine that understands requests written in plain language, and measure if it works.

## Time & cost budget

We expect you can complete this task in less than 3 hours. You have one week from the day you receive the email to send it back.

You will receive an OpenRouter API key from us along with this repository, so you can use a real model in the demo. Read it from the `OPENROUTER_API_KEY` environment variable and never commit it.

Please use the `openai/gpt-5.6-luna` model. You will have a small budget, so please be careful with your usage. Going over budget means that you're very likely doing something wrong. Reach out to us if you have questions about your usage.

## Using AI

**You are encouraged to use AI tools.** We want to make this as close to reality as possible, so it doesn't make sense to ask you not to use them. But be ready to explain what you did in a 60-75 minute technical deep dive:

- Why that embedding model, and what you traded off
- What text you put into the embedding, and why that and not something else
- What you tried and discarded
- What you would do differently with more time

We want to understand how you reasoned about the problem and your solution.

## The data

Everything is in `data/`:

| File | What it is |
|---|---|
| `recipes.parquet` | 30,000 recipes |
| `queries_dev.json` | Natural-language queries with graded relevance judgements |

`recipes.parquet` columns: `recipe_id`, `name`, `description`, `tags`, `ingredients`, `RecipeCategory`, `created_at`, `cooking_time`, `preparation_time`, `servings`, and nine nutrition fields (`calories`, `total_fat`, `saturated_fat`, `cholesterol`, `sodium`, `carbohydrates`, `fiber`, `sugar`, `protein`).

**`queries_dev.json`** is a list of objects shaped `{"text": <query>, "relevant": {recipe_id: grade}}`, where `2` means strongly relevant, `1` relevant, and anything absent is not relevant.

## The task

### a) Semantic retrieval

Embed the catalogue and retrieve against a natural-language query (or use lexical search if you prefer so!). You can start with a small model you can run locally, but you are free to use any embedding model you like.

If your hardware does not allow you to run a model locally, you can use the Qwen model: `qwen/qwen3-embedding-8b` on OpenRouter. The budget easily covers embedding the full catalogue, but please be careful with your usage. 

### b) Offline evaluation

Report `recall@10` and `ndcg@10` over `queries_dev.json`. You don't need to write any evaluation code: `evaluation.py` already defines the metrics and provides two functions you can run as-is:

- `evaluate_dev()` — scores against `data/queries_dev.json`, for your own iteration
- `evaluate_test()` — the same measurement against `data/queries_test.json`

Your only job is to make `search.py` work: the evaluation calls its `warmup()` once, then `search(query, k)` per query. A skeleton at the repository root shows both functions and the result shape we expect. Replace the placeholder bodies, keep the signatures, and `python evaluation.py dev` runs.

We will give you `queries_test.json` during the technical interview and ask you to run `evaluate_test()`, so make sure it works from a fresh checkout without any edits. It is a held-out set in the same format as the dev file.

We don't expect you to optimize for score. Use the numbers to guide your decisions and be ready to explain them.

In your notes, tell us how you interpret your numbers and how much you trust them. Bear in mind that you can see the dev judgements while you build, so a good dev score partly measures how hard you tuned against them.

### c) Query understanding

Turn a natural-language request into whatever structured form your retrieval needs (time limits, dietary constraints, ingredients on hand), and write a one-line reason for each recommendation.

To be clear about the scope: **we are not asking for a chatbot**. Your system gets a single request and produces a ranked list of recipes. You do not need to handle follow-up questions, clarifications, or multi-turn conversations.

```
Query: I have chicken, a lemon and half an hour
Response: <list of recipes>
```

**Using an LLM here is required**. Put the provider behind an interface so it can be swapped, and handle the ways it fails in practice: a missing key, a timeout, a malformed or empty response. We care about how you designed the interface (not UI) at least as much as about what the model produces.

## Deliverables

**1. A repository.** Structure, tests, error handling and dependency management are all your call.

**2. Notes (two pages, maximum).** Answer the following questions:

- High-level overview of your approach and design decisions
- Any interesting observations or insights from the data?
- What did you leave out?
- Evaluation results
- How do you make this production-ready? What are the risks in a production environment?
- What would you do differently with more time?

**3. Demo & technical deep dive.** Share your screen, walk us through your code, and explain your design decisions. A simple CLI script (query in, top K results out) is enough. Working on a polished UI won't improve your evaluation, and it's not a good use of your time.

## How to send it back

A private Git repository. Add `dylanjcastillo` and `idiaz01` as collaborators.
