import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage
import psycopg2
from config.settings import Settings as ConfigSettings
from config.logging_config import app_logger
from src.prompts import (
    SYSTEM_PROMPT, 
    REPORT_PROMPT_TEMPLATE, 
    REPORT_QUERY_GENERATION_PROMPT,
    CONVERSATION_SUMMARY_PROMPT
)

from utils.s3_manager import S3Manager
from llama_index.core.response_synthesizers import get_response_synthesizer

class RAG:
    def __init__(self):
        self.logger = app_logger
        self.index = None
        self.chat_engine = None
        self.query_engine = None
        self.initialization_error = None
        
        try:
            self.db_password = ConfigSettings.db_password

            self._initialize_db()
            
            self.llm = Bedrock(
                model=ConfigSettings.bedrock_llm_model,
                region_name=ConfigSettings.bedrock_region,
                temperature=ConfigSettings.temperature,
                timeout=ConfigSettings.request_timeout,
            )
            
            self.embed_model = BedrockEmbedding(
                model=ConfigSettings.bedrock_embed_model,
                region_name=ConfigSettings.bedrock_region,
            )
            
            self.vector_store = PGVectorStore.from_params(
                host=ConfigSettings.db_host,
                port=ConfigSettings.db_port,
                database=ConfigSettings.db_name,
                user=ConfigSettings.db_user,
                password=self.db_password,
                table_name=ConfigSettings.vector_store_table,
                embed_dim=1024,
            )
            
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter(
                chunk_size=ConfigSettings.chunk_size, 
                chunk_overlap=ConfigSettings.chunk_overlap
            )
            self.logger.info("RAG engine initialized successfully.")
        except Exception as e:
            self.logger.error(f"CRITICAL: Failed to initialize RAG engine: {e}")
            self.initialization_error = str(e)

    def _initialize_db(self):
        """Ensure the pgvector extension is installed."""
        try:
            conn = psycopg2.connect(
                host=ConfigSettings.db_host,
                port=ConfigSettings.db_port,
                database=ConfigSettings.db_name,
                user=ConfigSettings.db_user,
                password=self.db_password,
                sslmode=ConfigSettings.db_sslmode
            )
            conn.autocommit = True

            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.close()
            self.logger.info("Database extension 'vector' verified/installed.")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise e

    def _parse_retrieval_queries(self, llm_output: str) -> list[str]:
        lines = llm_output.strip().split("\n")
        queries = [
            line.strip().lstrip("-*").lstrip("12345.").strip()
            for line in lines
            if len(line.strip()) > 8
        ]
        return list(dict.fromkeys(queries))[:5]

    def create_index(self, documents, index_persist_dir=None):
        if self.initialization_error:
            raise ValueError(f"RAG Engine failed to initialize: {self.initialization_error}")
        if not documents:
            raise ValueError("Cannot create index from an empty list of documents.")

        self.logger.info(f"Creating/Loading index for {len(documents)} documents in PGVector.")
        self.index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=self.storage_context,
            show_progress=True
        )

        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            system_prompt=SYSTEM_PROMPT,
            similarity_top_k=ConfigSettings.similarity_top_k,
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=ConfigSettings.similarity_top_k,
        )

    def query(self, query, chat_history=None):
        if self.initialization_error:
            raise ValueError(f"RAG Engine failed to initialize: {self.initialization_error}")
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

    def _get_retrieval_queries(self, chat_history):
        limited_history = chat_history[-(ConfigSettings.history_turns * 2):]
        conversation_summary = "\n".join([f"- {m['role']}: {m['content']}" for m in limited_history])

        try:
            generation_prompt = REPORT_QUERY_GENERATION_PROMPT.format(conversation_summary=conversation_summary)
            query_gen_response = self.llm.complete(generation_prompt)
            query_gen_text = getattr(query_gen_response, "text", str(query_gen_response))
            
            retrieval_queries = self._parse_retrieval_queries(query_gen_text)
            if retrieval_queries:
                return retrieval_queries

        except Exception as exc:
            self.logger.warning(f"Failed to generate retrieval queries from LLM. Using fallback. Error: {exc}")

        return [
            "Principais requisitos funcionais e não funcionais",
            "Riscos técnicos e operacionais identificados",
            "Recomendações e ações sugeridas",
        ]

    def _get_evidence_nodes(self, retrieval_queries):
        retriever = self.index.as_retriever(
            similarity_top_k=min(max(ConfigSettings.similarity_top_k, 4), 5)
        )
        
        evidence_nodes = []
        seen_keys = set()
        max_evidence_nodes = 20

        for rq in retrieval_queries:
            try:
                retrieved = retriever.retrieve(rq)
                for node_with_score in retrieved:
                    node = getattr(node_with_score, "node", None)
                    if not node: continue
                    
                    key = node.node_id or node.get_content(metadata_mode="none")[:200]
                    if key not in seen_keys:
                        seen_keys.add(key)
                        evidence_nodes.append(node_with_score)
                    
                    if len(evidence_nodes) >= max_evidence_nodes:
                        break
            except Exception as exc:
                self.logger.warning(f"Failed to retrieve nodes for query '{rq}': {exc}")
            
            if len(evidence_nodes) >= max_evidence_nodes:
                break
        
        if not evidence_nodes:
            self.logger.warning("No evidence nodes found. Using fallback query.")
            return retriever.retrieve("Resumo técnico com requisitos, riscos e recomendações.")
            
        return evidence_nodes

    def generate_report_data(self, chat_history):
        if self.initialization_error:
            raise ValueError(f"RAG Engine failed to initialize: {self.initialization_error}")
        if not self.index:
            raise ValueError("Document index has not been created.")
        if not chat_history:
            raise ValueError("Chat history is empty.")

        self.logger.info("Generating final report data.")
        
        conversation_summary_text = self._get_conversation_summary(chat_history)
        self.logger.info("Conversation summary generated.")

        retrieval_queries = self._get_retrieval_queries(chat_history)
        self.logger.info(f"Report retrieval queries generated: {len(retrieval_queries)}")
        
        evidence_nodes = self._get_evidence_nodes(retrieval_queries)
        self.logger.info(f"Synthesizing report from {len(evidence_nodes)} evidence nodes.")

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            summary_template=PromptTemplate(REPORT_PROMPT_TEMPLATE),
            use_async=False,
        )

        report_objective = (
            "Gerar relatório final técnico, fiel aos documentos, cobrindo insights, "
            "requisitos, riscos e recomendações."
        )
        response = response_synthesizer.synthesize(
            query=report_objective,
            nodes=evidence_nodes,
        )
        
        insights_text = getattr(response, "response", str(response))
        self.logger.info("Report generation complete.")
        
        return {
            "summary": conversation_summary_text,
            "insights": insights_text,
        }

    def _get_conversation_summary(self, chat_history):
        limited_history = chat_history[-(ConfigSettings.history_turns * 2):]
        conversation_text = "\n".join([f"- {m['role']}: {m['content']}" for m in limited_history])
        
        prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation_summary=conversation_text)
        
        try:
            response = self.llm.complete(prompt)
            return getattr(response, "text", str(response))
        except Exception as e:
            self.logger.error(f"Failed to generate conversation summary: {e}")
            return "Não foi possível gerar o resumo da conversa."
