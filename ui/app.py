import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import os
import uuid
import hashlib
from llama_index.core import SimpleDirectoryReader
from config.settings import Settings as ConfigSettings
from config.logging_config import app_logger

from src.engine import RAG
from src.report import generate_technical_report
from utils.check_perfomance import check_performance
from utils.s3_manager import S3Manager


def _fingerprint_uploads(uploaded_files):
    """Creates a stable fingerprint for the set of uploaded files."""
    if not uploaded_files:
        return None
    
    file_details = sorted([(f.name, f.size) for f in uploaded_files])
    
    hasher = hashlib.sha256()
    for name, size in file_details:
        hasher.update(f"{name}:{size}".encode())
        
    return hasher.hexdigest()

def _build_upload_snapshot(uploaded_files):
    """Creates a snapshot of current file names, sizes, and content hashes."""
    if not uploaded_files:
        return {}
    
    snapshot = {}
    for f in uploaded_files:
        content = f.getvalue()
        content_hash = hashlib.sha256(content).hexdigest()
        snapshot[f.name] = {
            "size": f.size,
            "hash": content_hash
        }
    return snapshot

def initialize_session_state():
    """Initializes the Streamlit session state with default values."""
    defaults = {
        "session_id": str(uuid.uuid4()),
        "chat_history": [],
        "rag_manager": RAG(),
        "s3_manager": S3Manager(),
        "documents_ready": False,
        "indexed_doc_names": [],
        "uploaded_file_snapshots": {},
        "uploaded_files_fingerprint": None,
        "logger": app_logger,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.logger.info("Streamlit session initialized or reloaded.")


def handle_file_uploads():
    """Manages file uploads and triggers document processing."""
    st.header("⚙️ Configurações")
    
    provider = ConfigSettings.llm_provider.upper()
    provider_icon = "🤖" if provider == "GEMINI" else "☁️"
    st.caption(f"{provider_icon} Provedor Ativo: **{provider}**")
    
    uploaded_files = st.file_uploader(
        "Upload de Documentos Técnicos",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )

    current_fingerprint = _fingerprint_uploads(uploaded_files)
    if current_fingerprint != st.session_state.uploaded_files_fingerprint:
        st.session_state.documents_ready = False
        st.session_state.uploaded_files_fingerprint = current_fingerprint
        st.info("Novos documentos detectados. Clique em 'Processar' para indexá-los.")

    if st.button("🚀 Processar Documentos", disabled=not uploaded_files):
        process_documents(uploaded_files)

@check_performance
def process_documents(uploaded_files):
    """Handles the logic of processing and indexing uploaded documents."""
    st.session_state.logger.info("Processing documents button clicked.")
    if not uploaded_files:
        st.warning("Por favor, faça o upload de pelo menos um documento.")
        st.session_state.logger.warning("Process documents button clicked with no files uploaded.")
        return

    temp_dir = f"./data/temp_{st.session_state.session_id}"
    index_persist_dir = f"./data/index_cache_{st.session_state.uploaded_files_fingerprint}"

    with st.spinner("Indexando documentos... Por favor, aguarde."):
        st.session_state.logger.info("Starting document indexing.")
        os.makedirs(temp_dir, exist_ok=True)
        
        valid_files = []
        for f in uploaded_files:
            bytes_data = f.getvalue()
            with open(os.path.join(temp_dir, f.name), "wb") as buffer:
                buffer.write(bytes_data)
            
            if st.session_state.s3_manager.upload_bytes(bytes_data, f.name):
                valid_files.append(f)
            else:
                st.error(f"Falha ao enviar '{f.name}' para o S3. O arquivo será ignorado.")

        if not valid_files:
            st.warning("Nenhum arquivo pôde ser processado.")
            return

        try:
            snapshots = _build_upload_snapshot(valid_files)
            
            documents = SimpleDirectoryReader(temp_dir).load_data()
            final_docs = []
            for doc in documents:
                file_name = doc.metadata.get("file_name")
                if file_name in snapshots:
                    doc.metadata["file_hash"] = snapshots[file_name]["hash"]
                    final_docs.append(doc)

            st.session_state.rag_manager.create_index(final_docs, index_persist_dir)
            st.session_state.documents_ready = True
            st.session_state.indexed_doc_names = [f.name for f in uploaded_files]
            st.session_state.uploaded_file_snapshots = snapshots
            st.success("Documentos prontos para análise!")
            st.session_state.logger.info("Documenting indexing completed successfully.")
        except (ValueError, Exception) as e:
            st.error(f"Falha ao processar documentos: {e}")
            st.session_state.logger.error(f"Error processing documents: {e}", exc_info=True)

@check_performance
def handle_report_generation():
    """Handles the final report generation and download."""
    st.divider()
    if st.button("📄 Gerar Relatório Final", disabled=not st.session_state.chat_history):
        st.session_state.logger.info("Generate report button clicked.")
        with st.spinner("Sintetizando conclusões..."):
            try:
                report_data = st.session_state.rag_manager.generate_report_data(
                    st.session_state.chat_history,
                    document_snapshots=st.session_state.uploaded_file_snapshots
                )
                pdf_bytes = generate_technical_report(
                    report_data,
                    st.session_state.indexed_doc_names
                )
                
                st.download_button(
                    label="📥 Baixar Relatório PDF",
                    data=pdf_bytes,
                    file_name=f"analise_tecnica_{st.session_state.session_id[:8]}.pdf",
                    mime="application/pdf"
                )
                st.session_state.logger.info("Report generated and download button displayed.")
            except Exception as e:
                st.error(f"Falha ao gerar relatório: {e}")
                st.session_state.logger.error(f"Failed to generate report: {e}", exc_info=True)

def display_chat_history():
    """Displays the chat history in the main area."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

@check_performance
def handle_chat_input():
    """Handles user input and assistant response."""
    if prompt := st.chat_input("Ex: Quais os principais riscos deste projeto?"):
        st.session_state.logger.info(f"New user prompt: '{prompt[:50]}...'")
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.documents_ready:
                st.warning("Por favor, processe os documentos na barra lateral antes de fazer uma pergunta.")
                st.session_state.logger.warning("Query attempted before documents were processed.")
                st.stop()
                
            with st.spinner("Analisando..."):
                try:
                    response = st.session_state.rag_manager.query(prompt, st.session_state.chat_history)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.session_state.logger.info("Assistant response successfully generated.")
                except Exception as e:
                    st.error(f"Falha ao consultar o assistente: {e}")
                    st.session_state.logger.error(f"Failed to query the assistant: {e}", exc_info=True)

def apply_custom_css():
    """Injeta estilos CSS personalizados no streamlit"""
    st.markdown(
        """
        <style>
        /* Importação da fonte Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

        /* Esconder UI padrão do Streamlit (Decluttering) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Tipografia global */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Estilização dos Botões */
        .stButton > button {
            border-radius: 8px;
            transition: all 0.2s ease-in-out;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            filter: brightness(1.1);
        }

        /* Balões de Chat (Glassmorphism sutil) */
        .stChatMessage {
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Personalização do File Uploader */
        .stFileUploader > div > div {
            border-radius: 12px;
            border-style: dashed;
            border-color: #334155;
            background-color: rgba(30, 41, 59, 0.5);
            transition: border-color 0.2s ease;
        }
        .stFileUploader > div > div:hover {
            border-color: #0D9488;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Clarus", page_icon="🕸️", layout="wide")
    
    apply_custom_css()
    
    initialize_session_state()

    if st.session_state.rag_manager.initialization_error:
        st.error(f"⚠️ Erro Crítico na Inicialização: {st.session_state.rag_manager.initialization_error}")
        st.info("Verifique as credenciais do Banco de Dados e as permissões do Bedrock.")

    with st.sidebar:
        handle_file_uploads()
        handle_report_generation()


    st.title("🕸️ Clarus: Assistente de Análise Documental")
    st.caption("Faça upload dos seus documentos para extrair informações, cruzar dados com o histórico da base e gerar insights.")
    display_chat_history()
    handle_chat_input()

if __name__ == "__main__":
    main()