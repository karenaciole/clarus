import streamlit as st
import os
import sys
import uuid
from pathlib import Path
from llama_index.core import SimpleDirectoryReader

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.engine import RAG


st.set_page_config(page_title="Clarus", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = RAG()

temp_dir = f"./data/temp_{st.session_state.session_id}"

with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded_files = st.file_uploader(
        "Upload de Documentos Técnicos", 
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )
    
    if st.button("🚀 Processar Documentos") and uploaded_files:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        with st.spinner("Indexando documentos..."):
            for f in uploaded_files:
                with open(os.path.join(temp_dir, f.name), "wb") as buffer:
                    buffer.write(f.getbuffer())
            
            documents = SimpleDirectoryReader(temp_dir).load_data()
            st.session_state.rag_manager.create_index(documents)
            st.success("Documentos prontos para análise!")

    st.divider()
    if st.button("📄 Gerar Relatório Final"):
        if st.session_state.chat_history:
            with st.spinner("Sintetizando conclusões com Mistral..."):
                prompt_final = (
                    "Com base em toda a nossa conversa acima e nos documentos analisados, "
                    "elabore um texto estruturado contendo os 'Principais Insights' e as "
                    "'Conclusões Técnicas' finais."
                )
                insights_finais = st.session_state.rag_manager.query(prompt_final)
                
                doc_names = [f.name for f in uploaded_files] if uploaded_files else ["Documentos em memória"]
                
                from src.report import generate_technical_report
                pdf_bytes = generate_technical_report(
                    st.session_state.chat_history,
                    doc_names,
                    insights_finais
                )

                if isinstance(pdf_bytes, bytearray):
                    pdf_bytes = bytes(pdf_bytes)
                
                st.download_button(
                    label="📥 Baixar Relatório PDF",
                    data=pdf_bytes,
                    file_name=f"analise_tecnica_{st.session_state.session_id[:8]}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Inicie uma conversa antes de gerar o relatório.")

st.title("📑 Assistente de Análise Técnica")
st.caption("Especialista em Requisitos, Riscos e Recomendações")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Quais os principais riscos deste projeto?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            response = st.session_state.rag_manager.query(prompt)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})