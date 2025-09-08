# models.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedder:
    """Wrapper around SentenceTransformer to produce numpy vectors."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        # convert_to_numpy=True ensures we get numpy arrays directly
        return self.model.encode(texts, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True)

class FAISSIndex:
    """Simple FAISS wrapper using inner-product on L2-normalized vectors (cosine similarity)."""
    def __init__(self, dim):
        self.dim = dim
        # IndexFlatIP uses inner product; we normalize vectors to use it as cosine-sim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors):
        arr = np.asarray(vectors).astype('float32')
        # normalize before adding (works in-place)
        faiss.normalize_L2(arr)
        self.index.add(arr)

    def search(self, query_vector, top_k=10):
        q = np.asarray(query_vector).astype('float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        return I[0].tolist(), D[0].tolist()

class TFIDFFallback:
    """A tiny TF-IDF fallback search used when embeddings or OpenAI are not available."""
    def __init__(self, max_features=5000):
        self.vec = TfidfVectorizer(max_features=max_features)
        self.X = None
        self.docs = None

    def fit(self, docs):
        # docs: list[str]
        self.docs = list(docs)
        self.X = self.vec.fit_transform(self.docs)

    def query(self, q, top_k=5):
        if self.X is None:
            return [], []
        qv = self.vec.transform([q])
        scores = (self.X * qv.T).toarray().squeeze()
        idx = np.argsort(-scores)[:top_k]
        return idx.tolist(), scores[idx].tolist()
