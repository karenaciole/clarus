"""Criação dos modelos (LLM e embeddings) via LangChain.

Centraliza a escolha de provedor (Gemini ou AWS Bedrock) para que o restante
do código dependa apenas das interfaces padrão do LangChain
(`BaseChatModel` e `Embeddings`).
"""
import time
from typing import Callable, Optional

from botocore.config import Config
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from config.logging_config import app_logger
from config.settings import Settings as ConfigSettings


class ResilientEmbeddings(Embeddings):
    """Envolve um modelo de embeddings com tratamento de rate limit.

    - `delay_seconds`: pausa antes de cada lote (útil para cotas do Bedrock).
    - Se o provedor devolver menos vetores do que textos (rejeição parcial
      do lote), refaz a vetorização um a um, usando um vetor nulo apenas
      para textos que continuarem falhando.
    """

    def __init__(
        self,
        base: Embeddings,
        dimension: int,
        delay_seconds: float = 0.0,
        logger=app_logger,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base = base
        self.dimension = dimension
        self.delay_seconds = delay_seconds
        self.logger = logger
        self._sleep = sleep

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.delay_seconds:
            self._sleep(self.delay_seconds)

        try:
            vectors = self.base.embed_documents(texts)
        except Exception as e:  # noqa: BLE001 - qualquer falha do provedor dispara o fallback
            self.logger.warning(f"Falha no lote de embeddings ({e}). Fazendo fallback 1-a-1...")
            vectors = []

        if len(vectors) == len(texts):
            return vectors

        self.logger.warning(f"Lote retornou {len(vectors)}/{len(texts)} vetores. Fazendo fallback 1-a-1...")
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * self.dimension
        try:
            return self.base.embed_documents([text])[0]
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Erro no embedding 1-a-1: {e}")
            return [0.0] * self.dimension

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(text)


def resolve_embed_dim(provider: str) -> int:
    """Dimensão dos vetores no pgvector para o provedor ativo."""
    if provider == "gemini":
        return ConfigSettings.embed_dim
    # Titan Embed v2 aceita 256, 512 ou 1024 dimensões (1024 é o padrão).
    return ConfigSettings.embed_dim if ConfigSettings.embed_dim in (256, 512, 1024) else 1024


def build_models(provider: Optional[str] = None) -> tuple[BaseChatModel, Embeddings, int]:
    """Retorna (llm, embeddings, embed_dim) para o provedor configurado."""
    provider = (provider or ConfigSettings.llm_provider).lower()
    embed_dim = resolve_embed_dim(provider)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        if not ConfigSettings.google_api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada. Verifique o Secrets Manager.")

        app_logger.info(f"🤖 Inicializando provedor: Google Gemini ({ConfigSettings.gemini_llm_model})")
        llm = ChatGoogleGenerativeAI(
            model=ConfigSettings.gemini_llm_model,
            google_api_key=ConfigSettings.google_api_key,
            temperature=ConfigSettings.temperature,
            timeout=ConfigSettings.request_timeout,
        )
        # O LangChain já aplica os task types RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY.
        base_embeddings = GoogleGenerativeAIEmbeddings(
            model=ConfigSettings.gemini_embed_model,
            google_api_key=ConfigSettings.google_api_key,
        )
        embeddings = ResilientEmbeddings(base_embeddings, dimension=embed_dim)
        return llm, embeddings, embed_dim

    from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

    app_logger.info(f"☁️ Inicializando provedor: AWS Bedrock ({ConfigSettings.bedrock_llm_model})")
    aws_config = Config(
        region_name=ConfigSettings.aws_region,
        retries={"max_attempts": 20, "mode": "adaptive"},
        read_timeout=ConfigSettings.request_timeout,
    )
    # A API Converse dá suporte nativo a tool calling, usado pelo agente.
    llm = ChatBedrockConverse(
        model_id=ConfigSettings.bedrock_llm_model,
        region_name=ConfigSettings.aws_region,
        temperature=ConfigSettings.temperature,
        config=aws_config,
    )
    base_embeddings = BedrockEmbeddings(
        model_id=ConfigSettings.bedrock_embed_model,
        region_name=ConfigSettings.aws_region,
        config=aws_config,
    )
    embeddings = ResilientEmbeddings(base_embeddings, dimension=embed_dim, delay_seconds=1.5)
    return llm, embeddings, embed_dim
