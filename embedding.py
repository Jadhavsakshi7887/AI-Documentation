# embedding.py
from sentence_transformers import SentenceTransformer
from pathlib import Path
try:
    from config import EMBEDDING_MODEL, EMBEDDING_CACHE_FOLDER
except ImportError:
    # Fallback if config not available
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_CACHE_FOLDER = str(Path(__file__).parent / "huggingface_models")

embed_model = SentenceTransformer(
    EMBEDDING_MODEL,
    cache_folder=EMBEDDING_CACHE_FOLDER
)

class LocalEmbedding:
    def embed_documents(self, texts):
        return embed_model.encode(
            texts, show_progress_bar=False, convert_to_tensor=False
        ).tolist()
    
    def embed_query(self, text):
        return embed_model.encode(
            [text], convert_to_tensor=False
        ).tolist()[0]
