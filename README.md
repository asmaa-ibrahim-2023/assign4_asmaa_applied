# assign4_asmaa_applied
# CSAI 422 — Assignment 4: Retrieval-Augmented Generation with Advanced Techniques

End-to-end textual RAG pipeline for factual question answering on the PopQA benchmark.

## Pipeline Overview

```
Query → Query Expansion → Dense + BM25 Retrieval → RRF Fusion → Cross-Encoder Reranking → LLM Answer Generation → Self-Reflection
```

## Setup

### Requirements

- Python 3.10+
- Google Colab (recommended) or a local environment with GPU/CPU

### Install dependencies

```bash
pip install datasets sentence-transformers faiss-cpu rank_bm25 pandas numpy tqdm requests groq
```

### Groq API Key (for LLM generation)

Create a free account at [groq.com](https://console.groq.com), generate an API key, then run this in a notebook cell before the generation cells:

```python
import os
os.environ['GROQ_API_KEY'] = 'gsk_your_key_here'
```

If no key is provided, the pipeline falls back to a deterministic extractive answer using the PopQA gold answers for demonstration.

## Running the Notebook

1. Open `asmaa_assign4.ipynb` in Google Colab
1. Set your Groq API key (optional, see above)
1. Run all cells: **Runtime → Run all**

Total runtime is approximately 10–15 minutes on a standard Colab CPU instance.

## Repository Structure

```
├── asmaa_assign4.ipynb   # Main notebook (all parts)
└── README.md             # This file
```

## System Configurations

|System                   |Recall@1|Recall@5|MRR  |
|-------------------------|--------|--------|-----|
|Dense baseline           |0.27    |0.34    |0.295|
|Dense + query expansion  |0.28    |0.34    |0.303|
|Hybrid RRF               |0.26    |0.35    |0.296|
|Hybrid + reranker        |0.32    |0.36    |0.334|
|Final (+ self-reflection)|0.32    |0.36    |0.334|

## Models Used

|Component       |Model                                   |
|----------------|----------------------------------------|
|Embedder        |`sentence-transformers/all-MiniLM-L6-v2`|
|Reranker        |`cross-encoder/ms-marco-MiniLM-L-6-v2`  |
|LLM (generation)|`llama-3.1-8b-instant` via Groq API     |

## Dataset

[PopQA](https://huggingface.co/datasets/akariasai/PopQA) — 14,267 factual questions derived from Wikidata triples. Evaluation subset: 100 questions sampled with `random_state=42`.
