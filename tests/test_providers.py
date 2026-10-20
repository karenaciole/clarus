from langchain_core.embeddings import Embeddings

from src.providers import ResilientEmbeddings


class FlakyEmbeddings(Embeddings):
    """Devolve lotes incompletos e falha para textos marcados como 'ruim'."""

    def embed_documents(self, texts):
        if len(texts) > 1:
            return [[1.0, 1.0]]  # lote parcial
        if "ruim" in texts[0]:
            raise RuntimeError("rate limit")
        return [[float(len(texts[0])), 0.0]]

    def embed_query(self, text):
        return [9.0, 9.0]


def test_falls_back_one_by_one_when_batch_is_partial():
    sleeps = []
    emb = ResilientEmbeddings(FlakyEmbeddings(), dimension=2, delay_seconds=1.5, sleep=sleeps.append)
    vectors = emb.embed_documents(["abc", "ruim", "  "])
    assert vectors == [[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert sleeps == [1.5]


def test_query_and_empty_input():
    emb = ResilientEmbeddings(FlakyEmbeddings(), dimension=2)
    assert emb.embed_query("x") == [9.0, 9.0]
    assert emb.embed_documents([]) == []
