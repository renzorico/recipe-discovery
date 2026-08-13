# Recipe Discovery: Notes

## Approach and design decisions

The request is vague ("something light after training"), but the catalogue is literal (it contains names and ingredient lists). I bridge this gap in one step: an LLM (GPT-5.6-Luna) rewrites the request into descriptive keywords. I then embed these with All-MiniLM-L6-V2 and rank 30,000 recipes by dot-product similarity. Warmup() embeds and caches all vectors once. Search() rewrites, embeds, ranks and asks the LLM for a one-line explanation for each result. Every model call goes through a single LLMClient, so swapping providers only requires a change to one file. Any LLM failure (e.g. missing key, timeout, empty or malformed response) results in degraded but usable embedding-only results.

Two choices carried the weight. The first is what goes into each embedding: name, category, tags, ingredient names, total time and nutrition as labels rather than numbers, because models understand "low-calorie" but cannot reason that 250 \< 300\. Bucketing pays off twice: the raw fields are inaccurate (e.g. a recipe with 70,000 calories or 382,000 mg of sodium), and the labels absorb these outliers free of charge. I dropped the description: it has 29,670 distinct values, but each is the same template, "Make and share this \[NAME\] recipe from Food.com", so the only thing that varies is the name, which I already embed. The rewrite step is what makes plain language work. Without it, the response to "something light after a heavy training session" is a literal match: "Crystal Light Sunshine Punch". With it, the response is protein shakes and post-workout oatmeal, and the similarity score rises from \~0.3 to \~0.6.

## The data insight I care about most

Four dev queries look like corrupted labels. Three of them genuinely are. The "big bowl of pasta" query, for example, has 1,697 relevant recipes, none of which contain the words "pasta", "spaghetti" or "noodle", they are all desserts. Similarly, "an ice-cold drink" resolves to quick breads, and "a light fruity dessert" resolves to gumbo and stew. These queries score 0.0 no matter how good the retrieval is.

The fourth query, "the kitchen is packed up and the power is off", is not corrupted and I almost discarded it. Eighty per cent of its relevant recipes have a cooking time of zero, compared to a corpus baseline of 14 per cent; the espresso martini and agua fresca that initially seemed absurd are no-cook drinks. This is a deliberate test of indirection: no power means no cooking. Its sibling, "our cooktop is broken", sits at 2% no-cook because a broken cooktop still leaves you with an oven. The person who wrote these labels knew that a cooktop is not an oven. That is craft, not noise. Therefore, the real contamination rate is three out of eighteen, not four, and the lesson is that label noise and label sophistication appear identical at first glance: you have to analyse the relevant set against the corpus, rather than looking at three examples.

## Evaluation results

|  | All 18 queries | Corrupted removed |
| :---- | :---- | :---- |
| recall@10 | 0.389 | \~0.5 |
| ndcg@10 | 0.278 | \~0.36 |

Because every query has between 636 and 3,905 relevant recipes, the denominator of recall@10 is always 10 (min(k, len)), so recall@10 is precision@10 in disguise. The goal is to keep the top ten free of off-target items, not to find rare ones. I report all 18 as the headline figure, since production cannot delete its hard cases. I trust the ranking of these numbers over the absolutes, since development judgements were visible while building, and the real measure is the held-out evaluate\_test() (which runs unedited). I will rerun the cleaned figure live.

## What I left out, production risks and next steps

I retrieve the embeddings alone. The three gaps, in payoff order, also represent my next steps:

* Structured filters: parsing "under 30 minutes", "vegan" and "I have chicken" into hard constraints before ranking. This ensures that a no-cook query cannot return a two-hour braise.  
* Hybrid retrieval: BM25 is used for exact ingredient names, where dense vectors are weakest.  
* Cross-encoder re-ranking: this is performed over the top 100 for NDCG. The main risks in production are:  
* Latency: two LLM calls take 5–10 seconds, and cache rewrites and asynchronous explanations are required.  
* Reliability: retries with backoff and a circuit breaker are needed for the tested fallback.  
* Scale: the NumPy dot product fails past \~100k, so an ANN index should be used instead.  
* Prompt injection: input should be sanitised and output validated.  
* Silent decay: offline metrics miss drift, so click-through should be monitored online. If I had more time, I would also swap in a stronger embedder (Qwen/Qwen3-Embedding-8B) and add an A/B harness to evaluate strategies based on online behaviour rather than offline recall alone.

