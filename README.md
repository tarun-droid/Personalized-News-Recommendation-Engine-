# Personalized News Recommendation Engine (LLM + Vector Search)
Self Project — Jul '25

This repo contains a news recommendation engine that uses Sentence-Transformer embeddings + FAISS for retrieval, simple user profiling for personalization, and OpenAI GPT for summarization (configurable via `configs.yaml`).

## Features
- Real-time article retrieval using FAISS vector search
- Personalization with user profiling & preference-based ranking
- GPT-based summarization with fallback
- Streamlit app for interactive demo
- YAML configuration for easy switching (embeddings / TF-IDF / summarization toggle)

## Quickstart
1. Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Set your OpenAI API key (optional, for GPT summaries):

```bash
export OPENAI_API_KEY="sk-..."
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Files
- app.py — Streamlit UI
- models.py — embedding, index, retrieval utilities
- utils.py — data ingestion, profiling, summarization fallback
- configs.yaml — project configuration
- sample_data.csv — small example dataset
- requirements.txt — Python dependencies

## Notes
- This starter project is designed for small datasets and demo use. For production you should persist FAISS indexes to disk, handle incremental updates, and secure your OpenAI API key.
- If you hit GPU/installation issues with sentence-transformers, consider using a smaller model or running in an environment with PyTorch preinstalled.
