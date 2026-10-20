"""Motor RAG do Clarus, orquestrado com LangChain.

Fluxo:
1. Os documentos são quebrados em chunks e vetorizados no PostgreSQL/pgvector
   (`langchain_postgres.PGVector`), com o hash SHA-256 do arquivo nos metadados.
2. No chat, a pergunta é reescrita com base no histórico (condense question),
   os chunks são recuperados com filtro por hash (isolamento de contexto) e o
   LLM responde apenas com base nesse contexto.
3. No relatório, a conversa é resumida, sub-queries são geradas e os trechos
   recuperados de cada documento são sintetizados em árvore (map-reduce).
"""
import sys
from pathlib import Path
from urllib.parse import quote_plus

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: E402
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from config.logging_config import app_logger  # noqa: E402
from config.settings import Settings as ConfigSettings  # noqa: E402
from src.prompts import (  # noqa: E402
    CONDENSE_QUESTION_PROMPT,
    CONVERSATION_SUMMARY_PROMPT,
    HISTORICAL_CLASSIFICATION_PROMPT,
    HISTORICAL_SEARCH_NOTICE,
    REPORT_PROMPT_TEMPLATE,
    REPORT_QUERY_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)

# Aproximação usada para manter CHUNK_SIZE/CHUNK_OVERLAP em tokens, como antes.
CHARS_PER_TOKEN = 4
# Tamanho máximo de contexto enviado ao LLM em cada passo da síntese em árvore.
TREE_SUMMARIZE_MAX_CHARS = 12_000
MAX_REPORT_QUERIES = 3


# ---------------------------------------------------------------------------
# Funções auxiliares (puras, testáveis sem banco nem provedor de LLM)
# ---------------------------------------------------------------------------

def hash_filter(hashes: list[str] | str | None) -> dict | None:
    """Filtro de metadados do PGVector que restringe a busca a arquivos específicos."""
    if not hashes:
        return None
    if isinstance(hashes, str):
        return {"file_hash": {"$eq": hashes}}
    return {"file_hash": {"$in": list(hashes)}}


def to_lc_messages(chat_history: list[dict] | None, limit: int, current_query: str | None = None) -> list[BaseMessage]:
    """Converte o histórico do Streamlit em mensagens LangChain (últimos `limit` turnos)."""
    history = list(chat_history or [])
    # A UI adiciona a pergunta atual ao histórico antes de chamar o motor.
    if current_query and history and history[-1].get("role") == "user" and history[-1].get("content") == current_query:
        history = history[:-1]

    messages: list[BaseMessage] = []
    for m in history[-limit:] if limit > 0 else []:
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "assistant":
            messages.append(AIMessage(content=m["content"]))
    return messages


def format_docs(docs: list[Document]) -> str:
    """Formata os chunks recuperados indicando o arquivo de origem."""
    return "\n\n".join(
        f"[Fonte: {d.metadata.get('file_name', 'desconhecida')}]\n{d.page_content}" for d in docs
    )


def build_conversational_chain(llm: BaseChatModel, retriever: Runnable, system_prompt: str) -> Runnable:
    """Cadeia LCEL equivalente ao modo `condense_plus_context`.

    Entrada: {"question": str, "history": list[BaseMessage]}. Saída: str.
    """
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_QUESTION_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    def standalone_question(inputs: dict) -> str:
        if not inputs.get("history"):
            return inputs["question"]
        return condense_chain.invoke(inputs).strip() or inputs["question"]

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}\n\nContexto dos documentos:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]).partial(system_prompt=system_prompt)

    return (
        RunnablePassthrough.assign(standalone=RunnableLambda(standalone_question))
        | RunnablePassthrough.assign(context=(lambda x: x["standalone"]) | retriever | format_docs)
        | answer_prompt
        | llm
        | StrOutputParser()
    )


def parse_queries(text: str, max_queries: int = MAX_REPORT_QUERIES) -> list[str]:
    """Normaliza as sub-queries geradas pelo LLM: remove numeração, vazios e duplicatas."""
    queries: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        q = line.strip().lstrip("-*•0123456789.) ").strip().strip('"')
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    return queries[:max_queries]


def _group_by_size(texts: list[str], max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for t in texts:
        if current and size + len(t) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(t)
        size += len(t)
    if current:
        groups.append(current)
    return groups


def tree_summarize(
    llm: BaseChatModel,
    template: str,
    contexts: list[str],
    query_str: str,
    max_chars: int = TREE_SUMMARIZE_MAX_CHARS,
) -> str:
    """Síntese em árvore (map-reduce) que respeita a janela de contexto do LLM.

    Agrupa os trechos em lotes de até `max_chars`, sintetiza cada lote e repete
    sobre as sínteses parciais até restar uma única resposta.
    """
    if not contexts:
        return "Não foram encontrados trechos relevantes nos documentos para gerar insights."

    parser = StrOutputParser()
    level = list(contexts)
    while True:
        outputs = [
            parser.invoke(llm.invoke(template.format(context_str="\n\n".join(group), query_str=query_str)))
            for group in _group_by_size(level, max_chars)
        ]
        if len(outputs) == 1:
            return outputs[0]
        if len(_group_by_size(outputs, max_chars)) >= len(outputs):
            # As sínteses parciais já não cabem juntas; evita laço infinito.
            return "\n\n".join(outputs)
        level = outputs


def build_connection_url() -> str:
    """URL SQLAlchemy (driver psycopg 3) para o RDS PostgreSQL."""
    user = quote_plus(ConfigSettings.db_user or "")
    password = quote_plus(ConfigSettings.db_password or "")
    return (
        f"postgresql+psycopg://{user}:{password}@{ConfigSettings.db_host}:{ConfigSettings.db_port}/"
        f"{ConfigSettings.db_name}?sslmode={ConfigSettings.db_sslmode}"
    )


# ---------------------------------------------------------------------------
# Motor RAG
# ---------------------------------------------------------------------------

class RAG:
    def __init__(self):
        self.logger = app_logger
        self.ready = False
        self.current_hashes: list[str] = []
        self.initialization_error = None

        try:
            from langchain_postgres import PGVector
            from sqlalchemy import create_engine

            from src.providers import build_models

            provider = ConfigSettings.llm_provider.lower()
            self.llm, self.embeddings, embed_dim = build_models(provider)

            connection_url = build_connection_url()
            self._db_engine = create_engine(connection_url, pool_pre_ping=True)
            # create_extension=True garante a extensão `vector` no RDS.
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                connection=self._db_engine,
                collection_name=ConfigSettings.vector_store_table,
                embedding_length=embed_dim,
                use_jsonb=True,
                create_extension=True,
            )

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=ConfigSettings.chunk_size * CHARS_PER_TOKEN,
                chunk_overlap=ConfigSettings.chunk_overlap * CHARS_PER_TOKEN,
            )
            self.logger.info(f"RAG engine (LangChain) inicializado com sucesso (Provedor: {provider}).")

        except Exception as e:
            self.logger.error(f"ERRO CRÍTICO na inicialização do RAG: {e}")
            self.initialization_error = str(e)

    # -- Ingestão ---------------------------------------------------------

    def _get_indexed_hashes(self) -> set[str]:
        """Hashes de arquivos já indexados nesta coleção (deduplicação)."""
        from sqlalchemy import text

        sql = text(
            "SELECT DISTINCT e.cmetadata->>'file_hash' "
            "FROM langchain_pg_embedding e "
            "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
            "WHERE c.name = :collection AND e.cmetadata->>'file_hash' IS NOT NULL"
        )
        try:
            with self._db_engine.connect() as conn:
                return {row[0] for row in conn.execute(sql, {"collection": ConfigSettings.vector_store_table})}
        except Exception as e:
            self.logger.warning(f"Não foi possível recuperar hashes existentes: {e}")
            return set()

    def split_documents(self, documents: list[Document]) -> tuple[list[Document], list[str]]:
        """Quebra os documentos em chunks com IDs determinísticos por arquivo."""
        chunks = self.text_splitter.split_documents(documents)
        counters: dict[str, int] = {}
        ids = []
        for chunk in chunks:
            file_hash = chunk.metadata.get("file_hash", "sem-hash")
            counters[file_hash] = counters.get(file_hash, 0) + 1
            ids.append(f"{file_hash}-{counters[file_hash]}")
        return chunks, ids

    def create_index(self, documents: list[Document], index_persist_dir=None):
        """Indexa apenas documentos novos (deduplicação por hash) e define o escopo da sessão.

        `index_persist_dir` é mantido por compatibilidade: o índice vive no pgvector.
        """
        if self.initialization_error:
            raise ValueError(f"O motor RAG falhou ao iniciar: {self.initialization_error}")
        if not documents:
            raise ValueError("Lista de documentos vazia.")

        existing_hashes = self._get_indexed_hashes()
        new_docs = [doc for doc in documents if doc.metadata.get("file_hash") not in existing_hashes]

        if not new_docs:
            self.logger.info("♻️ Todos os documentos já estão no banco. Pulando chamadas de API de Embedding.")
        else:
            chunks, ids = self.split_documents(new_docs)
            self.logger.info(f"🆕 Indexando {len(new_docs)} novos documentos ({len(chunks)} chunks).")
            batch_size = 10
            for start in range(0, len(chunks), batch_size):
                self.vector_store.add_documents(chunks[start:start + batch_size], ids=ids[start:start + batch_size])

        self.current_hashes = sorted({doc.metadata["file_hash"] for doc in documents if doc.metadata.get("file_hash")})
        self.ready = True

    # -- Consulta ---------------------------------------------------------

    def get_retriever(self, hashes: list[str] | str | None = None, k: int | None = None):
        search_kwargs = {"k": k or ConfigSettings.similarity_top_k}
        filters = hash_filter(hashes)
        if filters:
            search_kwargs["filter"] = filters
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def retrieve(self, query: str, historical: bool = False, k: int | None = None) -> list[Document]:
        """Busca chunks relevantes: na sessão atual ou em todo o histórico."""
        if not self.ready:
            raise ValueError("Índice não inicializado. Processe os documentos primeiro.")
        hashes = None if historical else self.current_hashes
        return self.get_retriever(hashes, k).invoke(query)

    def is_historical_query(self, query: str) -> bool:
        try:
            resp = self.llm.invoke(HISTORICAL_CLASSIFICATION_PROMPT.format(query=query))
            return "YES" in StrOutputParser().invoke(resp).upper()
        except Exception as e:
            self.logger.warning(f"Erro na classificação da query, assumindo falso: {e}")
            return False

    def query(self, query, chat_history=None):
        """Realiza uma pergunta ao assistente usando o contexto dos documentos."""
        if not self.ready:
            raise ValueError("Índice não inicializado. Processe os documentos primeiro.")

        is_historical = self.is_historical_query(query)
        system_prompt = SYSTEM_PROMPT
        if is_historical:
            system_prompt += HISTORICAL_SEARCH_NOTICE
            retriever = self.get_retriever(None)
            self.logger.info("Query classificada como busca histórica (sem filtros).")
        else:
            retriever = self.get_retriever(self.current_hashes)
            self.logger.info("Query classificada como busca na sessão atual (com filtros).")

        chain = build_conversational_chain(self.llm, retriever, system_prompt)
        history = to_lc_messages(chat_history, ConfigSettings.history_turns, current_query=query)
        response_text = chain.invoke({"question": query, "history": history})

        self.logger.info(f"Resposta gerada ({len(response_text)} caracteres).")
        return response_text

    # -- Relatório --------------------------------------------------------

    def _collect_contexts(self, queries: list[str], hashes) -> list[str]:
        retriever = self.get_retriever(hashes)
        seen: set[str] = set()
        contexts: list[str] = []
        for q in queries:
            for doc in retriever.invoke(q):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    contexts.append(format_docs([doc]))
        return contexts

    def generate_report_data(self, chat_history, document_snapshots=None):
        """Sintetiza as conclusões da conversa em um relatório estruturado por documento."""
        if not self.ready:
            raise ValueError("Índice não disponível para gerar relatório.")

        self.logger.info("Gerando dados para o relatório final...")
        parser = StrOutputParser()

        conversation_text = "\n".join(f"- {m['role']}: {m['content']}" for m in chat_history[-10:])
        summary_text = parser.invoke(
            self.llm.invoke(CONVERSATION_SUMMARY_PROMPT.format(conversation_summary=conversation_text))
        )

        try:
            queries_text = parser.invoke(
                self.llm.invoke(REPORT_QUERY_GENERATION_PROMPT.format(conversation_summary=summary_text))
            )
            search_queries = parse_queries(queries_text) or [summary_text]
        except Exception as e:
            self.logger.warning(f"Falha ao gerar sub-queries, usando o resumo: {e}")
            search_queries = [summary_text]
        query_str = "\n".join(search_queries)

        insights_per_doc = {}
        if document_snapshots:
            for file_name, file_info in document_snapshots.items():
                contexts = self._collect_contexts(search_queries, file_info["hash"])
                insights_per_doc[file_name] = tree_summarize(self.llm, REPORT_PROMPT_TEMPLATE, contexts, query_str)
        else:
            contexts = self._collect_contexts(search_queries, self.current_hashes)
            insights_per_doc["Documentos Analisados"] = tree_summarize(
                self.llm, REPORT_PROMPT_TEMPLATE, contexts, query_str
            )

        return {
            "summary": summary_text,
            "insights_per_doc": insights_per_doc,
        }
