# utils.py
import os
import pandas as pd
import numpy as np
import yaml
import openai
from sklearn.preprocessing import normalize

# Load configuration
with open('configs.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

def load_data(path='sample_data.csv'):
    df = pd.read_csv(path)
    if 'title' not in df.columns or 'content' not in df.columns:
        raise ValueError("sample_data.csv must contain 'title' and 'content' columns")
    df['text'] = df['title'].fillna('') + ' -- ' + df['content'].fillna('')
    return df

def build_user_profile(interactions_df, embedder):
    """Build a single vector profile from a DataFrame of interactions (must contain 'text')."""
    if interactions_df is None or interactions_df.empty:
        return None
    texts = interactions_df['text'].astype(str).tolist()
    vecs = embedder.encode(texts)
    profile = np.mean(vecs, axis=0)
    profile = normalize(profile.reshape(1, -1))[0]
    return profile

def gpt_summarize(text, max_tokens=None, model=None):
    """Summarize text using OpenAI ChatCompletion API. Falls back to first 200 chars if no API key or on error."""
    max_tokens = max_tokens or CONFIG.get('openai', {}).get('max_tokens', 120)
    model = model or CONFIG.get('openai', {}).get('model', 'gpt-4o-mini')
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return text[:200] + ('...' if len(text) > 200 else '')
    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"Summarize concisely the following article:\n\n{text}"}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        summary = resp['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        # graceful fallback
        return text[:200] + ('...' if len(text) > 200 else '')

def rank_results(df, profile_vector, recency_weight=0.4, personalization_weight=0.6):
    """Rank a small DataFrame of retrieved results by combining cosine-sim with recency."""
    if profile_vector is None:
        return df
    scores = []
    now = pd.Timestamp.now()
    for _, row in df.iterrows():
        emb = np.asarray(row['embedding'])
        if np.linalg.norm(emb) == 0 or np.linalg.norm(profile_vector) == 0:
            sim = 0.0
        else:
            sim = float(np.dot(emb / np.linalg.norm(emb), profile_vector / np.linalg.norm(profile_vector)))
        # recency score (higher for recent)
        try:
            days = (now - pd.to_datetime(row.get('published'))).days
            recency_score = 1 / (1 + days / 30) if days >= 0 else 1.0
        except Exception:
            recency_score = 0.5
        score = personalization_weight * sim + recency_weight * recency_score
        scores.append(score)
    out = df.copy()
    out['score'] = scores
    return out.sort_values('score', ascending=False)

def openai_available():
    return bool(os.environ.get('OPENAI_API_KEY'))
