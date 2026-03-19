from fpdf import FPDF
from datetime import datetime

class Report(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 12)
        self.cell(0, 10, "Relatório de Análise Técnica - Assistente RAG", border=False, ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", align="C")

def generate_technical_report(chat_history, documents, insights, filename="relatorio_tecnico.pdf"):
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Resumo da Análise de Documentos", ln=True)
    
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 7, f"Data de Emissão: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.cell(0, 7, f"Documentos Analisados: {', '.join(documents)}", ln=True)
    pdf.ln(10)

    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Perguntas e Respostas Realizadas", ln=True)
    pdf.ln(2)
    
    for i, chat in enumerate(chat_history):
        pdf.set_font("helvetica", "B", 10)
        pdf.multi_cell(0, 7, f"Pergunta {i+1}: {chat['content']}" if chat['role'] == 'user' else "")
        
        if chat['role'] == 'assistant':
            pdf.set_font("helvetica", "", 10)
            pdf.multi_cell(0, 6, f"Resposta: {chat['content']}")
            pdf.ln(4)

    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Principais Insights e Conclusões", ln=True)
    pdf.ln(2)
    
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 6, insights)

    return pdf.output()