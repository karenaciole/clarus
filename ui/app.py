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
    """Creates a snapshot of current file names and sizes."""
    if not uploaded_files:
        return {}
    return {f.name: f.size for f in uploaded_files}

st.set_page_config(page_title="Clarus", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = RAG()

if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False

if "indexed_doc_names" not in st.session_state:
    st.session_state.indexed_doc_names = []

if "uploaded_file_snapshots" not in st.session_state:
    st.session_state.uploaded_file_snapshots = {}

if "uploaded_files_fingerprint" not in st.session_state:
    st.session_state.uploaded_files_fingerprint = None

if "logger" not in st.session_state:
    st.session_state.logger = app_logger
    st.session_state.logger.info("Streamlit session started.")

with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded_files = st.file_uploader(
        "Upload de Documentos Técnicos",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )

    current_fingerprint = _fingerprint_uploads(uploaded_files)
    if current_fingerprint != st.session_state.uploaded_files_fingerprint:
        st.session_state.documents_ready = False
        st.session_state.uploaded_files_fingerprint = current_fingerprint

    def process_documents():
        st.session_state.logger.info("Processing documents button clicked.")
        if not uploaded_files:
            st.warning("Por favor, faça o upload de pelo menos um documento.")
            st.session_state.logger.warning("Process documents button clicked with no files uploaded.")
            return

        temp_dir = f"./data/temp_{st.session_state.session_id}"
        index_persist_dir = f"./data/index_cache_{st.session_state.uploaded_files_fingerprint}"

        with st.spinner("Indexando documentos... Por favor, aguarde."):
            st.session_state.logger.info("Starting document indexing.")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            for f in uploaded_files:
                with open(os.path.join(temp_dir, f.name), "wb") as buffer:
                    buffer.write(f.getbuffer())
            
            try:
                documents = SimpleDirectoryReader(temp_dir).load_data()
                st.session_state.rag_manager.create_index(documents, index_persist_dir)
                st.session_state.documents_ready = True
                st.session_state.indexed_doc_names = [f.name for f in uploaded_files]
                st.session_state.uploaded_file_snapshots = _build_upload_snapshot(uploaded_files)
                st.success("Documentos prontos para análise!")
                st.session_state.logger.info("Document indexing completed successfully.")
            except ValueError as e:
                st.error(f"Falha ao processar documentos: {e}")
                st.session_state.logger.error(f"Error processing documents: {e}", exc_info=True)
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}")
                st.session_state.logger.error(f"An unexpected error occurred during document processing: {e}", exc_info=True)

    if st.button("🚀 Processar Documentos"):
        process_documents()

    st.divider()
    if st.button("📄 Gerar Relatório Final"):
        st.session_state.logger.info("Generate report button clicked.")
        if st.session_state.chat_history:
            with st.spinner("Sintetizando conclusões..."):
                try:
                    insights_finais = st.session_state.rag_manager.generate_report_query(
                        st.session_state.chat_history
                    )
                    
                    from src.report import generate_technical_report
                    pdf_bytes = generate_technical_report(
                        st.session_state.chat_history,
                        st.session_state.indexed_doc_names,
                        insights_finais
                    )
                    
                    st.download_button(
                        label="📥 Baixar Relatório PDF",
                        data=bytes(pdf_bytes),
                        file_name=f"analise_tecnica_{st.session_state.session_id[:8]}.pdf",
                        mime="application/pdf"
                    )
                    st.session_state.logger.info("Report generated and download button displayed.")
                except Exception as e:
                    st.error(f"Falha ao gerar relatório: {e}")
                    st.session_state.logger.error(f"Failed to generate report: {e}", exc_info=True)
        else:
            st.warning("Inicie uma conversa antes de gerar o relatório.")
            st.session_state.logger.warning("Generate report button clicked with no chat history.")

st.title("📑 Assistente de Análise Técnica")
st.caption("Especialista em Requisitos, Riscos e Recomendações")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
                st.session_state.logger.info("Assistant response successfully generated and displayed.")
            except Exception as e:
                st.error(f"Falha ao consultar o assistente: {e}")
                st.session_state.logger.error(f"Failed to query the assistant: {e}", exc_info=True)