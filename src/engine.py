import os
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage
from config.settings import Settings as ConfigSettings
from config.logging_config import app_logger

from llama_index.core.response_synthesizers import get_response_synthesizer

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
        self.logger = app_logger
        self.logger.info("RAG engine initialized.")
        self.logger.info(f"LLM Provider: {ConfigSettings.llm_provider}")
        if ConfigSettings.llm_provider == "ollama":
            self.logger.info(f"Ollama Model: {ConfigSettings.ollama_model_name}")
        else:
            self.logger.info(f"Gemini Model: {ConfigSettings.gemini_model_name}")
        self.logger.info(f"Embedding Model: {ConfigSettings.embedding_model_name}")
        self.logger.info(f"Persistence enabled: {ConfigSettings.persist_index}")

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
            "Você é o Clarus, um assistente de análise técnica de documentos. Sua comunicação deve ser sempre em Português (Brasil).\n\n"
            "**Sua Missão Principal:**\n"
            "Responder às perguntas do usuário de forma precisa e objetiva, baseando-se **exclusivamente** no conteúdo dos documentos fornecidos como contexto. Não utilize conhecimento prévio ou informações externas.\n\n"
            "**Diretrizes de Resposta:**\n"
            "1. **Fidelidade ao Contexto**: Se a resposta para uma pergunta não estiver nos documentos, afirme claramente: 'Com base nos documentos fornecidos, não encontrei informações sobre este tópico.'\n"
            "2. **Clareza e Organização**: Apresente as respostas de forma clara. Use listas (bullet points) para detalhar informações e **negrito** para destacar termos e conceitos importantes.\n"
            "3. **Tom Profissional**: Mantenha um tom profissional e direto, focando em fornecer informações úteis e precisas."
        )

    def _build_report_prompt_template(self):
        return (
            "Você é um assistente técnico sênior, especializado em compilar relatórios de análise detalhados. "
            "Sua tarefa é sintetizar a conversa e os documentos fornecidos em um relatório final estruturado.\n\n"
            "Use APENAS as informações presentes no CONTEXTO recuperado abaixo.\n"
            "Se alguma informação não estiver no contexto, declare explicitamente que não há evidência suficiente.\n\n"
            "Objetivo da síntese:\n{query_str}\n\n"
            "CONTEXTO RECUPERADO:\n{context_str}\n\n"
            "Com base no contexto acima, elabore um texto estruturado contendo:\n"
            "1.  **Principais Insights**: Um resumo dos pontos mais importantes discutidos.\n"
            "2.  **Identificação de Requisitos**: Liste todos os requisitos funcionais e não funcionais mencionados.\n"
            "3.  **Análise de Riscos**: Identifique e descreva os potenciais riscos técnicos, operacionais e de negócio.\n"
            "4.  **Sugestão de Recomendações**: Proponha recomendações claras e acionáveis para mitigar os riscos e atender aos requisitos.\n\n"
            "**Formato da Resposta**: Apresente a resposta de forma organizada, utilizando seções distintas para cada um dos pontos acima.\n"
            "**Idioma**: Responda sempre em Português (Brasil), com um tom profissional e direto.\n"
            "**Fidelidade ao Contexto**: Baseie-se exclusivamente nas informações contidas no histórico da conversa e nos documentos."
        )

    def _build_report_query_generation_prompt(self, conversation_summary):
        return (
            "Você receberá um histórico de conversa sobre análise técnica de documentos.\n"
            "Gere entre 3 e 5 consultas curtas e objetivas para recuperar evidências dos documentos.\n"
            "As consultas devem cobrir: requisitos, riscos, decisões, dependências, restrições e recomendações.\n"
            "Retorne somente uma lista, uma consulta por linha, sem explicações.\n\n"
            f"Histórico:\n{conversation_summary}"
        )

    def _parse_retrieval_queries(self, llm_output):
        queries = []
        for raw_line in llm_output.splitlines():
            line = raw_line.strip().lstrip("-*").strip()
            if not line:
                continue
            if line[0].isdigit() and "." in line:
                first_dot = line.find(".")
                if first_dot > 0:
                    line = line[first_dot + 1 :].strip()
            if len(line) >= 8:
                queries.append(line)
        unique = list(dict.fromkeys(queries))
        return unique[:5]

    def create_index(self, documents, index_persist_dir):
        if not documents:
            raise ValueError("Cannot create index from an empty list of documents.")

        self.logger.info(f"Creating index from {len(documents)} documents.")
        if ConfigSettings.persist_index and os.path.exists(index_persist_dir):
            self.logger.info(f"Loading existing index from: {index_persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=index_persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            self.logger.info("Building new index.")
            self.index = VectorStoreIndex.from_documents(documents)
            if ConfigSettings.persist_index:
                self.logger.info(f"Persisting index to: {index_persist_dir}")
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
        
        if hasattr(self.chat_engine, "chat"):
            response = self.chat_engine.chat(query, chat_history=limited_history)
        else:
            response = self.chat_engine.query(query) 

        response_text = getattr(response, "response", str(response))
        self.logger.info(f"Query response: '{response_text[:100]}...'")
        return response_text

    def generate_report_query(self, chat_history):
        if not self.index:
            raise ValueError("Document index has not been created. Please create the index before querying.")
        if not chat_history:
            raise ValueError("Chat history is empty. Start a conversation before generating the report.")

        self.logger.info("Generating final report.")

        report_prompt_template = PromptTemplate(self._build_report_prompt_template())

        limited_history = chat_history[-(ConfigSettings.history_turns * 2):]
        conversation_summary = "\n".join([f"- {m['role']}: {m['content']}" for m in limited_history])

        retrieval_queries = []
        try:
            generation_prompt = self._build_report_query_generation_prompt(conversation_summary)
            query_gen_response = self.llm.complete(generation_prompt)
            query_gen_text = getattr(query_gen_response, "text", str(query_gen_response))
            retrieval_queries = self._parse_retrieval_queries(query_gen_text)
        except Exception as exc:
            self.logger.warning(f"Failed to generate retrieval queries from LLM. Using fallback queries. Error: {exc}")

        if not retrieval_queries:
            retrieval_queries = [
                "Principais requisitos funcionais e não funcionais",
                "Riscos técnicos e operacionais identificados",
                "Recomendações e ações sugeridas",
            ]

        self.logger.info(f"Report retrieval queries generated: {len(retrieval_queries)}")

        retriever = self.index.as_retriever(
            similarity_top_k=min(max(ConfigSettings.similarity_top_k, 4), 5)
        )
        evidence_nodes = []
        seen_keys = set()
        max_evidence_nodes = 20
        for rq in retrieval_queries:
            try:
                nodes = retriever.retrieve(rq)
            except Exception as exc:
                self.logger.warning(f"Failed to retrieve nodes for query '{rq}': {exc}")
                continue

            for node_with_score in nodes:
                node = getattr(node_with_score, "node", None)
                node_id = getattr(node, "node_id", None)
                text = ""
                if node is not None and hasattr(node, "get_content"):
                    text = node.get_content(metadata_mode="none")[:200]
                key = node_id or text
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    evidence_nodes.append(node_with_score)
                if len(evidence_nodes) >= max_evidence_nodes:
                    break
            if len(evidence_nodes) >= max_evidence_nodes:
                break

        if not evidence_nodes:
            self.logger.warning("No evidence nodes found with generated retrieval queries. Using fallback query.")
            evidence_nodes = retriever.retrieve("Resumo técnico com requisitos, riscos e recomendações.")

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            summary_template=report_prompt_template,
            use_async=False,
        )

        report_objective = (
            "Gerar relatório final técnico, fiel aos documentos, cobrindo insights, "
            "requisitos, riscos e recomendações."
        )
        self.logger.info(f"Synthesizing report from {len(evidence_nodes)} evidence nodes.")
        response = response_synthesizer.synthesize(
            query=report_objective,
            nodes=evidence_nodes,
        )
        
        response_text = getattr(response, "response", str(response))
        self.logger.info("Report generation complete.")
        return response_text
