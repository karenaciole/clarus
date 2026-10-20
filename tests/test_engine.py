from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.engine import (
    build_conversational_chain,
    format_docs,
    hash_filter,
    parse_queries,
    to_lc_messages,
    tree_summarize,
)


def test_hash_filter():
    assert hash_filter(None) is None
    assert hash_filter([]) is None
    assert hash_filter("abc") == {"file_hash": {"$eq": "abc"}}
    assert hash_filter(["a", "b"]) == {"file_hash": {"$in": ["a", "b"]}}


def test_to_lc_messages_drops_current_query_and_limits_turns():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
    ]
    messages = to_lc_messages(history, limit=3, current_query="q3")
    assert [type(m) for m in messages] == [AIMessage, HumanMessage, AIMessage]
    assert [m.content for m in messages] == ["a1", "q2", "a2"]
    assert to_lc_messages(None, limit=5) == []


def test_format_docs_cites_source():
    text = format_docs([Document(page_content="conteúdo", metadata={"file_name": "spec.pdf"})])
    assert text == "[Fonte: spec.pdf]\nconteúdo"


def test_parse_queries_normalizes_and_dedupes():
    raw = "1. riscos do projeto\n- Riscos do projeto\n\n* prazos de entrega\n\"orçamento\"\nextra"
    assert parse_queries(raw) == ["riscos do projeto", "prazos de entrega", "orçamento"]


def test_conversational_chain_condenses_with_history():
    llm = FakeListChatModel(responses=["Qual o prazo do projeto Alfa?", "O prazo é março."])
    seen_queries = []

    def fake_retriever(q):
        seen_queries.append(q)
        return [Document(page_content="Prazo: março", metadata={"file_name": "alfa.pdf"})]

    chain = build_conversational_chain(llm, RunnableLambda(fake_retriever), "Sistema")
    history = [HumanMessage(content="Fale do projeto Alfa"), AIMessage(content="É um projeto X.")]
    answer = chain.invoke({"question": "E o prazo?", "history": history})

    assert answer == "O prazo é março."
    assert seen_queries == ["Qual o prazo do projeto Alfa?"]


def test_conversational_chain_skips_condense_without_history():
    llm = FakeListChatModel(responses=["Resposta direta."])
    seen_queries = []
    chain = build_conversational_chain(
        llm, RunnableLambda(lambda q: seen_queries.append(q) or []), "Sistema"
    )
    assert chain.invoke({"question": "Pergunta?", "history": []}) == "Resposta direta."
    assert seen_queries == ["Pergunta?"]


def test_tree_summarize_reduces_in_levels():
    llm = FakeListChatModel(responses=["parcial 1", "parcial 2", "final"])
    template = "CTX:{context_str} Q:{query_str}"
    result = tree_summarize(llm, template, ["a" * 60, "b" * 60], "q", max_chars=100)
    assert result == "final"


def test_tree_summarize_without_context():
    llm = FakeListChatModel(responses=["nunca chamado"])
    assert "Não foram encontrados" in tree_summarize(llm, "{context_str}{query_str}", [], "q")
