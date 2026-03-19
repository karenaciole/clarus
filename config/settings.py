import dotenv
import os

dotenv.load_dotenv()

class Settings:
    model_name = os.getenv("DEFAULT_MODEL_NAME", "mistral")
    embedding_model_name = os.getenv(
        "DEFAULT_EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    request_timeout = float(os.getenv("DEFAULT_REQUEST_TIMEOUT", "120.0"))