import os
import sys
import time
import psycopg2
import json
from pathlib import Path
from botocore.config import Config

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
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

from config.settings import Settings as ConfigSettings
from config.logging_config import app_logger
from src.prompts import (
    SYSTEM_PROMPT, 
    REPORT_PROMPT_TEMPLATE, 
    REPORT_QUERY_GENERATION_PROMPT,
    CONVERSATION_SUMMARY_PROMPT
)

class RAG:
    def __init__(self):
        self.logger = app_logger
        self.index = None
        self.initialization_error = None
        
        try:
            self.db_password = ConfigSettings.db_password
            self._initialize_db()
            
            provider = ConfigSettings.llm_provider.lower()
            
            if provider == "gemini":
                self.logger.info(f"🤖 Inicializando Provedor Moderno: GoogleGenAI ({ConfigSettings.gemini_llm_model})")
                if not ConfigSettings.google_api_key:
                    raise ValueError("GOOGLE_API_KEY não encontrada. Verifique o Secrets Manager.")
                
                self.llm = GoogleGenAI(
                    model=ConfigSettings.gemini_llm_model,
                    api_key=ConfigSettings.google_api_key,
                    temperature=ConfigSettings.temperature,
                )
                self.embed_model = GoogleGenAIEmbedding(
                    model_name=ConfigSettings.gemini_embed_model,
                    api_key=ConfigSettings.google_api_key,
                )
                embed_dim = ConfigSettings.embed_dim
                
                original_gemini_get_text_embeddings = self.embed_model._get_text_embeddings
                def patched_gemini_get_text_embeddings(texts):
                    formatted_texts = [f"title: none | text: {t}" if t.strip() else t for t in texts]
                    
                    res = original_gemini_get_text_embeddings(formatted_texts)
                    if len(res) < len(texts):
                        self.logger.warning(f"Gemini: Batch retornou {len(res)}/{len(texts)}. Fazendo fallback 1-a-1...")
                        res = []
                        for t in formatted_texts:
                            try:
                                if not t.strip():
                                    res.append([0.0] * embed_dim)
                                else:
                                    res.append(self.embed_model.get_text_embedding(t))
                            except Exception as e:
                                self.logger.error(f"Erro no embedding 1-a-1: {e}")
                                res.append([0.0] * embed_dim)
                    return res
                self.embed_model._get_text_embeddings = patched_gemini_get_text_embeddings

                original_get_query_embedding = self.embed_model._get_query_embedding
                def patched_get_query_embedding(query):
                    formatted_query = f"task: search result | query: {query}"
                    return original_get_query_embedding(formatted_query)
                self.embed_model._get_query_embedding = patched_get_query_embedding

            
            else:
                self.logger.info("☁️ Inicializando Provedor: AWS Bedrock")
                aws_config = Config(
                    region_name=ConfigSettings.aws_region,
                    retries={'max_attempts': 20, 'mode': 'adaptive'}
                )

                self.llm = Bedrock(
                    model=ConfigSettings.bedrock_llm_model,
                    region_name=ConfigSettings.aws_region,
                    temperature=ConfigSettings.temperature,
                    timeout=ConfigSettings.request_timeout,
                    config=aws_config
                )
                self.embed_model = BedrockEmbedding(
                    model=ConfigSettings.bedrock_embed_model,
                    region_name=ConfigSettings.aws_region,
                    config=aws_config
                )
                
                original_get_text_embeddings = self.embed_model._get_text_embeddings
                def delayed_get_text_embeddings(texts):
                    self.logger.info(f"Bedrock: Processando {len(texts)} textos com delay de segurança...")
                    time.sleep(1.5)
                    return original_get_text_embeddings(texts)
                self.embed_model._get_text_embeddings = delayed_get_text_embeddings
                
                embed_dim = ConfigSettings.embed_dim if ConfigSettings.embed_dim != 768 else 1024

            self.vector_store = PGVectorStore.from_params(
                host=ConfigSettings.db_host,
                port=ConfigSettings.db_port,
                database=ConfigSettings.db_name,
                user=ConfigSettings.db_user,
                password=self.db_password,
                table_name=ConfigSettings.vector_store_table,
                embed_dim=embed_dim,
            )
            
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter(
                chunk_size=ConfigSettings.chunk_size, 
                chunk_overlap=ConfigSettings.chunk_overlap
            )
            self.logger.info(f"RAG engine inicializado com sucesso (Provedor: {provider}).")
            
        except Exception as e:
            self.logger.error(f"ERRO CRÍTICO na inicialização do RAG: {e}")
            self.initialization_error = str(e)

    def _initialize_db(self):
        """Garante que a extensão pgvector esteja instalada no RDS."""
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
            self.logger.info("Extensão 'vector' verificada/instalada no banco de dados.")
        except Exception as e:
            self.logger.error(f"Falha ao conectar ou inicializar o banco: {e}")
            raise e

    def _get_indexed_hashes(self):
        """Busca no banco de dados os hashes de arquivos que já foram indexados."""
        try:
            conn = psycopg2.connect(
                host=ConfigSettings.db_host,
                port=ConfigSettings.db_port,
                database=ConfigSettings.db_name,
                user=ConfigSettings.db_user,
                password=self.db_password,
                sslmode=ConfigSettings.db_sslmode
            )
            hashes = set()
            with conn.cursor() as cur:
                cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{ConfigSettings.vector_store_table}');")
                if cur.fetchone()[0]:
                    cur.execute(f"SELECT DISTINCT (metadata_->>'file_hash') FROM {ConfigSettings.vector_store_table} WHERE metadata_->>'file_hash' IS NOT NULL;")
                    rows = cur.fetchall()
                    hashes = {r[0] for r in rows}
            conn.close()
            return hashes
        except Exception as e:
            self.logger.warning(f"Não foi possível recuperar hashes existentes: {e}")
            return set()

    def create_index(self, documents, index_persist_dir=None):
        """Cria ou carrega o índice, realizando deduplicação inteligente."""
        if self.initialization_error:
            raise ValueError(f"O motor RAG falhou ao iniciar: {self.initialization_error}")
        if not documents:
            raise ValueError("Lista de documentos vazia.")
        existing_hashes = self._get_indexed_hashes()
        new_docs = [doc for doc in documents if doc.metadata.get("file_hash") not in existing_hashes]

        if not new_docs:
            self.logger.info("♻️ Todos os documentos já estão no banco. Pulando chamadas de API de Embedding.")
        else:
            self.logger.info(f"🆕 Indexando {len(new_docs)} novos documentos.")
            self.index = VectorStoreIndex.from_documents(
                new_docs, 
                storage_context=self.storage_context,
                show_progress=True,
                num_workers=1,
                embed_batch_size=10
            )
        
        if not self.index:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
            
        self.current_hashes = list(set([doc.metadata.get("file_hash") for doc in documents if doc.metadata.get("file_hash")]))

    def query(self, query, chat_history=None):
        """Realiza uma pergunta ao assistente usando o contexto dos documentos."""
        if not self.index:
            raise ValueError("Índice não inicializado. Processe os documentos primeiro.")
        
        classification_prompt = (
            "You are a helpful assistant classifying user queries.\n"
            "The user is currently analyzing some recently uploaded documents.\n"
            "Does the following query explicitly ask to search or reference 'old', 'previous', 'historical', or 'past' documents that are not in the current session?\n"
            "Reply with exactly 'YES' or 'NO'.\n\n"
            f"Query: {query}"
        )
        try:
            resp = self.llm.complete(classification_prompt)
            is_historical = "YES" in str(resp).upper()
        except Exception as e:
            self.logger.warning(f"Erro na classificação da query, assumindo falso: {e}")
            is_historical = False
        
        history_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in chat_history]
        limited_history = history_messages[-ConfigSettings.history_turns:] if history_messages else []
        
        
        filters = None
        system_prompt = SYSTEM_PROMPT
        
        if is_historical:
            system_prompt += "\n\nIMPORTANTE: O usuário perguntou explicitamente sobre documentos antigos/históricos. Você tem acesso a toda a base de dados. Ao responder, DEIXE CLARO que você precisou buscar no histórico e que está referenciando informações de documentos antigos/históricos armazenados no banco de dados."
            self.logger.info("Query classificada como busca histórica (sem filtros).")
        else:
            if hasattr(self, 'current_hashes') and self.current_hashes:
                filters = MetadataFilters(
                    filters=[MetadataFilter(key="file_hash", value=self.current_hashes, operator=FilterOperator.IN)]
                )
            self.logger.info("Query classificada como busca na sessão atual (com filtros).")

        temp_chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            system_prompt=system_prompt,
            similarity_top_k=ConfigSettings.similarity_top_k,
            filters=filters
        )
        
        response = temp_chat_engine.chat(query, chat_history=limited_history)
        response_text = getattr(response, "response", str(response))
        
        self.logger.info(f"Resposta gerada ({len(response_text)} caracteres).")
        return response_text

    def generate_report_data(self, chat_history, document_snapshots=None):
        """Sintetiza as conclusões da conversa em um relatório estruturado por documento."""
        if not self.index:
            raise ValueError("Índice não disponível para gerar relatório.")

        self.logger.info("Gerando dados para o relatório final...")
        
        conversation_text = "\n".join([f"- {m['role']}: {m['content']}" for m in chat_history[-10:]])
        summary_prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation_summary=conversation_text)
        summary_resp = self.llm.complete(summary_prompt)
        summary_text = getattr(summary_resp, "text", str(summary_resp))
        
        query_gen_prompt = REPORT_QUERY_GENERATION_PROMPT.format(conversation_summary=summary_text)
        queries_resp = self.llm.complete(query_gen_prompt)
        search_queries = getattr(queries_resp, "text", str(queries_resp))

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            summary_template=PromptTemplate(REPORT_PROMPT_TEMPLATE),
        )
        
        insights_per_doc = {}
        if document_snapshots:
            for file_name, file_info in document_snapshots.items():
                doc_hash = file_info["hash"]
                filters = MetadataFilters(
                    filters=[MetadataFilter(key="file_hash", value=doc_hash, operator=FilterOperator.EQ)]
                )
                query_engine = self.index.as_query_engine(
                    response_synthesizer=response_synthesizer,
                    filters=filters
                )
                insights_resp = query_engine.query(search_queries)
                insights_per_doc[file_name] = getattr(insights_resp, "response", str(insights_resp))
        else:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="file_hash", value=self.current_hashes, operator=FilterOperator.IN)]
            ) if hasattr(self, 'current_hashes') and self.current_hashes else None
            
            query_engine = self.index.as_query_engine(
                response_synthesizer=response_synthesizer,
                filters=filters
            )
            insights_resp = query_engine.query(search_queries)
            insights_per_doc["Documentos Analisados"] = getattr(insights_resp, "response", str(insights_resp))

        return {
            "summary": summary_text,
            "insights_per_doc": insights_per_doc,
        }
