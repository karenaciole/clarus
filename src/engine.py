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
        self.chat_engine = None
        self.query_engine = None
        self.initialization_error = None
        
        try:
            self.db_password = ConfigSettings.db_password
            self._initialize_db()
            
            # --- Configuração Dinâmica do Provedor de IA ---
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
                # A dimensão agora é acessada diretamente do atributo simplificado
                embed_dim = ConfigSettings.embed_dim
                
                # Freio de Segurança / Workaround para Gemini (Evita KeyError de batch incompleto)
                original_gemini_get_text_embeddings = self.embed_model._get_text_embeddings
                def patched_gemini_get_text_embeddings(texts):
                    res = original_gemini_get_text_embeddings(texts)
                    if len(res) < len(texts):
                        self.logger.warning(f"Gemini: Batch retornou {len(res)}/{len(texts)}. Fazendo fallback 1-a-1...")
                        res = []
                        for t in texts:
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
                
                # Freio de Segurança para Bedrock (Monkey-patching)
                original_get_text_embeddings = self.embed_model._get_text_embeddings
                def delayed_get_text_embeddings(texts):
                    self.logger.info(f"Bedrock: Processando {len(texts)} textos com delay de segurança...")
                    time.sleep(1.5)
                    return original_get_text_embeddings(texts)
                self.embed_model._get_text_embeddings = delayed_get_text_embeddings
                
                # Para Bedrock v2, se o usuário não configurou EMBED_DIM no secret, 
                # garantimos 1024 como fallback lógico aqui, ou usamos o valor da classe.
                embed_dim = ConfigSettings.embed_dim if ConfigSettings.embed_dim != 768 else 1024

            # --- Conexão com o Vector Store ---
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
            
            # Configurações Globais do LlamaIndex
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

        # 1. Deduplicação por Hash (Economia de Créditos)
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
        
        # 2. Garante que o índice aponte para o Vector Store (novo ou existente)
        if not self.index:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )

        # 3. Inicializa os motores de chat e consulta
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            system_prompt=SYSTEM_PROMPT,
            similarity_top_k=ConfigSettings.similarity_top_k,
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=ConfigSettings.similarity_top_k,
        )

    def query(self, query, chat_history=None):
        """Realiza uma pergunta ao assistente usando o contexto dos documentos."""
        if not self.chat_engine:
            raise ValueError("Índice não inicializado. Processe os documentos primeiro.")
        
        history_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in chat_history]
        limited_history = history_messages[-ConfigSettings.history_turns:] if history_messages else []
        
        response = self.chat_engine.chat(query, chat_history=limited_history)
        response_text = getattr(response, "response", str(response))
        
        self.logger.info(f"Resposta gerada ({len(response_text)} caracteres).")
        return response_text

    def generate_report_data(self, chat_history):
        """Sintetiza as conclusões da conversa em um relatório estruturado."""
        if not self.index:
            raise ValueError("Índice não disponível para gerar relatório.")

        self.logger.info("Gerando dados para o relatório final...")
        
        # Resumo da conversa
        conversation_text = "\n".join([f"- {m['role']}: {m['content']}" for m in chat_history[-10:]])
        summary_prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation_summary=conversation_text)
        summary_resp = self.llm.complete(summary_prompt)
        summary_text = getattr(summary_resp, "text", str(summary_resp))

        # Recuperação de Insights (Query Técnica)
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            summary_template=PromptTemplate(REPORT_PROMPT_TEMPLATE),
        )
        
        query_engine = self.index.as_query_engine(response_synthesizer=response_synthesizer)
        insights_resp = query_engine.query("Gere um relatório técnico com requisitos, riscos e recomendações.")
        insights_text = getattr(insights_resp, "response", str(insights_resp))

        return {
            "summary": summary_text,
            "insights": insights_text,
        }
