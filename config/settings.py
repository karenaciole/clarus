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

if _secret_id:
    try:
        _session = boto3.Session()
        _client = _session.client("secretsmanager", region_name=_region)
        _response = _client.get_secret_value(SecretId=_secret_id)
        _config_data = json.loads(_response["SecretString"])
        print(f"INFO: Configurações carregadas com sucesso do Secret: {_secret_id}")
    except Exception as e:
        print(f"AVISO: Não foi possível carregar o segredo '{_secret_id}'. Erro: {e}")

def _resolve_config(key, default=None):
    return _config_data.get(key, _get_env(key, default))

class Settings:
    """
    Gerenciador de Configurações resolvido.
    """
    llm_provider = _resolve_config("LLM_PROVIDER", "bedrock")
    bedrock_region = _resolve_config("AWS_REGION", "us-east-1")
    
    bedrock_llm_model = _resolve_config("BEDROCK_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
    bedrock_embed_model = _resolve_config("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    
    # Database
    db_host = _resolve_config("DB_HOST")
    db_port = int(_resolve_config("DB_PORT", "5432"))
    db_name = _resolve_config("DB_NAME", "clarus_db")
    db_user = _resolve_config("DB_USER", "clarus_admin")
    db_password = _resolve_config("DB_PASSWORD", "")
    db_sslmode = _resolve_config("DB_SSLMODE", "require")
    db_secret_arn = _resolve_config("DB_SECRET_ARN")

    if db_secret_arn and not db_password:
        try:
            _client = boto3.client("secretsmanager", region_name=_region)
            _resp = _client.get_secret_value(SecretId=db_secret_arn)
            _secret = json.loads(_resp["SecretString"])
            db_password = _secret.get("password", _secret.get("DB_PASSWORD", ""))
        except Exception:
            pass

    vector_store_table = _resolve_config("VECTOR_STORE_TABLE", "data_clarus")

    # S3
    s3_bucket_name = _resolve_config("S3_BUCKET_NAME", "clarus-bucket")
    s3_prefix = _resolve_config("S3_PREFIX", "users/")
    s3_kms_key_arn = _resolve_config("S3_KMS_KEY_ARN")

    # RAG Configs
    chunk_size = int(_resolve_config("CHUNK_SIZE", "512"))
    chunk_overlap = int(_resolve_config("CHUNK_OVERLAP", "128"))
    similarity_top_k = int(_resolve_config("SIMILARITY_TOP_K", "5"))
    similarity_cutoff = float(_resolve_config("SIMILARITY_CUTOFF", "0.0"))
    history_turns = int(_resolve_config("HISTORY_TURNS", "5"))
    
    temperature = float(_resolve_config("DEFAULT_TEMPERATURE", "0.1"))
    request_timeout = float(_resolve_config("DEFAULT_REQUEST_TIMEOUT", "120.0"))
    persist_index = str(_resolve_config("PERSIST_INDEX", "true")).lower() in ("true", "1", "yes")