import dotenv
import os

dotenv.load_dotenv()

def _to_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")

class Settings:
    # Core
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    
    # Ollama
    ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen3:8b")
    
    # Gemini
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

    # Embedding
    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-m3",
    )
    
    # RAG
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "128"))
    similarity_top_k = int(os.getenv("SIMILARITY_TOP_K", "5"))
    similarity_cutoff = float(os.getenv("SIMILARITY_CUTOFF", "0.7"))
    history_turns = int(os.getenv("HISTORY_TURNS", "5"))
    persist_index = _to_bool(os.getenv("PERSIST_INDEX", "True"))

    # LLM
    temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    request_timeout = float(os.getenv("DEFAULT_REQUEST_TIMEOUT", "120.0"))