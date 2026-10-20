"""Teste de integração com PostgreSQL + pgvector reais (modelos falsos).

Roda apenas quando CLARUS_TEST_DB_HOST está definido (no CI, um serviço
`pgvector/pgvector` é criado para isso).
"""
import os
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

pytestmark = pytest.mark.skipif(
    not os.getenv("CLARUS_TEST_DB_HOST"), reason="defina CLARUS_TEST_DB_HOST para rodar contra o pgvector"
)

DIM = 32


@pytest.fixture()
def rag(monkeypatch):
    from config.settings import Settings
    from src import engine, providers

    monkeypatch.setattr(Settings, "db_host", os.environ["CLARUS_TEST_DB_HOST"])
    monkeypatch.setattr(Settings, "db_port", int(os.getenv("CLARUS_TEST_DB_PORT", "5432")))
    monkeypatch.setattr(Settings, "db_name", os.getenv("CLARUS_TEST_DB_NAME", "clarus_db"))
    monkeypatch.setattr(Settings, "db_user", os.getenv("CLARUS_TEST_DB_USER", "clarus"))
    monkeypatch.setattr(Settings, "db_password", os.getenv("CLARUS_TEST_DB_PASSWORD", ""))
    monkeypatch.setattr(Settings, "db_sslmode", "disable")
    monkeypatch.setattr(Settings, "vector_store_table", f"test_{uuid.uuid4().hex[:8]}")
    monkeypatch.setattr(Settings, "similarity_top_k", 3)

    llm = FakeListChatModel(responses=["NO", "Resposta com base no contexto."])
    embeddings = DeterministicFakeEmbedding(size=DIM)
    monkeypatch.setattr(providers, "build_models", lambda provider=None: (llm, embeddings, DIM))

    rag = engine.RAG()
    assert rag.initialization_error is None, rag.initialization_error
    yield rag
    rag.vector_store.delete_collection()


def _doc(text, file_hash, name):
    return Document(page_content=text, metadata={"file_hash": file_hash, "file_name": name})


def test_index_dedup_and_session_isolation(rag):
    old = _doc("Relatório antigo sobre turbinas.", "hash-antigo", "antigo.pdf")
    rag.create_index([old])
    assert rag._get_indexed_hashes() == {"hash-antigo"}

    new = _doc("Especificação nova sobre freios.", "hash-novo", "novo.pdf")
    rag.create_index([old, new])  # o antigo não é reindexado
    assert rag._get_indexed_hashes() == {"hash-antigo", "hash-novo"}

    rag.current_hashes = ["hash-novo"]
    session_docs = rag.retrieve("turbinas", historical=False)
    assert {d.metadata["file_name"] for d in session_docs} == {"novo.pdf"}

    history_docs = rag.retrieve("turbinas", historical=True)
    assert {d.metadata["file_name"] for d in history_docs} == {"antigo.pdf", "novo.pdf"}


def test_query_end_to_end(rag):
    rag.create_index([_doc("O prazo do projeto é março.", "h1", "plano.txt")])
    assert rag.query("Qual o prazo?", chat_history=[]) == "Resposta com base no contexto."
