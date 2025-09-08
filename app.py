# app.py - Streamlit UI for the personalized news recommender
import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml
from models import Embedder, FAISSIndex, TFIDFFallback
from utils import load_data, build_user_profile, gpt_summarize, rank_results, openai_available

st.set_page_config(layout='wide', page_title='Personalized News Recommender')

with open('configs.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# caches
@st.cache_resource
def init_embedder():
    return Embedder(CONFIG.get('embedding_model', 'all-MiniLM-L6-v2'))

@st.cache_data
def prepare_index(df, embedder):
    texts = df['text'].astype(str).tolist()
    vecs = embedder.encode(texts)
    if vecs is None or len(vecs) == 0:
        raise RuntimeError('Embedding generation failed')
    idx = FAISSIndex(vecs.shape[1])
    idx.add(vecs)
    df2 = df.copy()
    df2['embedding'] = list(vecs)
    return idx, df2

st.title('Personalized News Recommendation Engine')

st.sidebar.header('Data')
uploaded = st.sidebar.file_uploader('Upload CSV (must have title, content columns)', type=['csv'])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error('Failed to read uploaded CSV: ' + str(e))
        st.stop()
else:
    df = load_data('sample_data.csv')

# create outputs dir if missing
os.makedirs('outputs', exist_ok=True)

embedder = init_embedder()
try:
    index, df_indexed = prepare_index(df, embedder)
except Exception as e:
    st.error('Failed to build index: ' + str(e))
    st.stop()

st.sidebar.header('User')
user_pref = st.sidebar.text_area('Enter a few keywords or interests (comma separated)', value='AI, climate, markets')
use_profile = st.sidebar.checkbox('Use profile-based ranking', value=True)

query = st.text_input('Search query', value='latest AI')
if st.button('Search'):
    if not query or str(query).strip() == '':
        st.warning('Enter a search query.')
    else:
        qvec = embedder.encode([query])[0]
        ids, dists = index.search(qvec, top_k=CONFIG.get('top_k_retrieval', 10))
        # filter invalid ids
        ids = [i for i in ids if i is not None and i >= 0 and i < len(df_indexed)]
        if len(ids) == 0:
            st.info('No results found. Try a broader query.')
        else:
            results = df_indexed.iloc[ids].copy().reset_index(drop=True)
            # personalization
            profile_vec = None
            if use_profile:
                prefs = [p.strip() for p in str(user_pref).split(',') if p.strip()]
                if prefs:
                    pseudo = pd.DataFrame({'text': prefs, 'published': pd.Timestamp.now()})
                    profile_vec = build_user_profile(pseudo, embedder)
                    results = rank_results(results, profile_vec, recency_weight=CONFIG.get('ranking', {}).get('recency_weight', 0.4), personalization_weight=CONFIG.get('ranking', {}).get('personalization_weight', 0.6))

            # summaries
            openai_ok = openai_available() and CONFIG.get('openai', {}).get('summarization', True)
            st.write('### Results')
            for _, row in results.head(20).iterrows():
                st.subheader(row.get('title', 'Untitled'))
                c1, c2 = st.columns([4,1])
                with c1:
                    st.write(row.get('content', ''))
                    if openai_ok:
                        summary = gpt_summarize(row.get('content', ''), max_tokens=CONFIG.get('openai', {}).get('max_tokens', 120), model=CONFIG.get('openai', {}).get('model', 'gpt-4o-mini'))
                    else:
                        summary = row.get('content', '')[:200] + ('...' if len(str(row.get('content', ''))) > 200 else '')
                    st.markdown(f"**Summary:** {summary}")
                with c2:
                    st.write('Published: ' + str(row.get('published', '')))
                    st.write('Source: ' + str(row.get('source', '')))
                    if 'score' in row:
                        st.write('Score: ' + str(round(float(row.get('score', 0)), 3)))
st.sidebar.markdown('---')
st.sidebar.write('Config')
st.sidebar.json(CONFIG)
