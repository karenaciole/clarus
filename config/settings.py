import os
import boto3
import json
import dotenv
from pathlib import Path

forbidden_profiles = ["", "clarus-dev", "default"]
if os.getenv("AWS_PROFILE") in forbidden_profiles:
    os.environ.pop("AWS_PROFILE", None)

project_root = Path(__file__).resolve().parents[1]

env_path = project_root / ".env"
if env_path.exists():
    dotenv.load_dotenv(dotenv_path=env_path, override=False)

def _get_env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    return val.strip().strip("\"").strip("'")

_config_data = {}
_secret_id = _get_env("CONFIG_SECRET_ID")
_region = _get_env("AWS_REGION", "us-east-1")

from config.logging_config import app_logger

if _secret_id:
    try:
        _session = boto3.Session()
        _client = _session.client("secretsmanager", region_name=_region)
        _response = _client.get_secret_value(SecretId=_secret_id)
        _config_data = json.loads(_response["SecretString"])
        app_logger.info(f"Configurações carregadas com sucesso do Secret: {_secret_id}")
    except Exception as e:
        app_logger.warning(f"Não foi possível carregar o segredo '{_secret_id}'. Erro: {e}")

def _resolve(key: str, default: any = None):
    """Auxiliar interno para resolver valor com prioridade: Secret > Environment > Default."""
    return _config_data.get(key, _get_env(key, default))

class Settings:
    """
    Configurações Centralizadas da Aplicação.
    Acesso direto via atributos de classe (ex: Settings.llm_provider).
    """
    llm_provider = _resolve("LLM_PROVIDER", "bedrock")
    aws_region = _resolve("AWS_REGION", "us-east-1")
    
    bedrock_llm_model = _resolve("BEDROCK_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
    bedrock_embed_model = _resolve("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    
    google_api_key = _resolve("GOOGLE_API_KEY")
    gemini_llm_model = _resolve("GEMINI_LLM_MODEL", "models/gemini-1.5-flash-lite")
    gemini_embed_model = _resolve("GEMINI_EMBED_MODEL", "models/text-embedding-004")
    
    db_host = _resolve("DB_HOST")
    db_port = int(_resolve("DB_PORT", "5432"))
    db_name = _resolve("DB_NAME", "clarus_db")
    db_user = _resolve("DB_USER", "clarus_admin")
    db_password = _resolve("DB_PASSWORD", "")
    db_sslmode = _resolve("DB_SSLMODE", "require")
    db_secret_arn = _resolve("DB_SECRET_ARN")
    vector_store_table = _resolve("VECTOR_STORE_TABLE", "data_clarus")
    
    embed_dim = int(_resolve("EMBED_DIM", "3072"))

    s3_bucket_name = _resolve("S3_BUCKET_NAME", "clarus-bucket")
    s3_prefix = _resolve("S3_PREFIX", "users/")
    s3_kms_key_arn = _resolve("S3_KMS_KEY_ARN")

    chunk_size = int(_resolve("CHUNK_SIZE", "512"))
    chunk_overlap = int(_resolve("CHUNK_OVERLAP", "128"))
    similarity_top_k = int(_resolve("SIMILARITY_TOP_K", "5"))
    similarity_cutoff = float(_resolve("SIMILARITY_CUTOFF", "0.0"))
    history_turns = int(_resolve("HISTORY_TURNS", "5"))
    
    temperature = float(_resolve("DEFAULT_TEMPERATURE", "0.1"))
    request_timeout = float(_resolve("DEFAULT_REQUEST_TIMEOUT", "120.0"))
    persist_index = str(_resolve("PERSIST_INDEX", "true")).lower() in ("true", "1", "yes")

    if db_secret_arn and not db_password:
        try:
            _client = boto3.client("secretsmanager", region_name=aws_region)
            _resp = _client.get_secret_value(SecretId=db_secret_arn)
            _sec = json.loads(_resp["SecretString"])
            db_password = _sec.get("password", _sec.get("DB_PASSWORD", ""))
        except Exception:
            pass
