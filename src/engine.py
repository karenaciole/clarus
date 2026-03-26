import os
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage
from config.settings import Settings as ConfigSettings

_EMBEDDING_CACHE = {}

def _get_embed_model(model_name):
    if model_name not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[model_name] = HuggingFaceEmbedding(model_name=model_name)
    return _EMBEDDING_CACHE[model_name]

class RAG:
    def __init__(self):
        self.llm = self._build_llm()
        self.embed_model = _get_embed_model(ConfigSettings.embedding_model_name)
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = SentenceSplitter(
            chunk_size=ConfigSettings.chunk_size, 
            chunk_overlap=ConfigSettings.chunk_overlap
        )
        
        self.index = None
        self.chat_engine = None
        self.query_engine = None

    def _build_llm(self):
        provider = ConfigSettings.llm_provider.lower()
        if provider == "ollama":
            return self._build_ollama_llm()
        elif provider == "gemini":
            return Gemini(
                model_name=ConfigSettings.gemini_model_name,
                temperature=ConfigSettings.temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _build_ollama_llm(self):
        return Ollama(
            model=ConfigSettings.ollama_model_name,
            temperature=ConfigSettings.temperature,
            request_timeout=ConfigSettings.request_timeout,
        )

    def _build_system_prompt(self):
        return (
            "Você é um assistente técnico altamente qualificado, especializado em análise de documentos. "
            "Sua principal função é extrair, analisar e estruturar informações críticas. "
            "Ao analisar o contexto fornecido, siga estritamente estas diretrizes:\n"
            "1.  **Identificação de Requisitos**: Liste todos os requisitos funcionais e não funcionais mencionados.\n"
            "2.  **Análise de Riscos**: Identifique e descreva os potenciais riscos técnicos, operacionais e de negócio.\n"
            "3.  **Sugestão de Recomendações**: Proponha recomendações claras e acionáveis para mitigar os riscos e atender aos requisitos.\n"
            "4.  **Formato da Resposta**: Apresente a resposta de forma organizada, utilizando seções distintas para 'Requisitos', 'Riscos' e 'Recomendações'.\n"
            "5.  **Idioma**: Responda sempre em Português (Brasil), com um tom profissional e direto.\n"
            "6.  **Fidelidade ao Contexto**: Baseie-se exclusivamente nas informações contidas nos documentos fornecidos."
        )

    def create_index(self, documents, index_persist_dir):
        if not documents:
            raise ValueError("Cannot create index from an empty list of documents.")

        if ConfigSettings.persist_index and os.path.exists(index_persist_dir):
            storage_context = StorageContext.from_defaults(persist_dir=index_persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            self.index = VectorStoreIndex.from_documents(documents)
            if ConfigSettings.persist_index:
                self.index.storage_context.persist(persist_dir=index_persist_dir)

        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            system_prompt=self._build_system_prompt(),
            similarity_top_k=ConfigSettings.similarity_top_k,
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=ConfigSettings.similarity_top_k,
        )

    def query(self, query, chat_history=None):
        if not self.chat_engine:
            raise ValueError("Document index has not been created. Please create the index before querying.")
        
        history_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in chat_history]
        limited_history = history_messages[-ConfigSettings.history_turns:] if history_messages else []
        
        # The CondensePlusContextChatEngine uses the .chat() method, not .query()
        if hasattr(self.chat_engine, "chat"):
            response = self.chat_engine.chat(query, chat_history=limited_history)
        else:
            response = self.chat_engine.query(query) # Fallback for other engine types

        return getattr(response, "response", str(response))

