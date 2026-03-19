from fpdf import FPDF
from datetime import datetime

class Report(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 12)
        self.cell(0, 10, "Relatório de Análise Técnica feito pelo Assistente Inteligente", border=False, ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", align="C")

def generate_technical_report(chat_history, documents, insights, filename="relatorio_tecnico.pdf"):
    def _to_text(value):
        if value is None:
            return ""
        return getattr(value, "response", str(value))

    def _write_multiline(pdf_obj, text, line_height=6):
        content_width = pdf_obj.w - pdf_obj.l_margin - pdf_obj.r_margin
        pdf_obj.set_x(pdf_obj.l_margin)
        pdf_obj.multi_cell(content_width, line_height, _to_text(text))

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
        content = _to_text(chat.get('content', ''))
        role = chat.get('role')

        if role == 'user':
            pdf.set_font("helvetica", "B", 10)
            _write_multiline(pdf, f"Pergunta {i+1}: {content}", line_height=7)

        if role == 'assistant':
            pdf.set_font("helvetica", "", 10)
            _write_multiline(pdf, f"Resposta: {content}", line_height=6)
            pdf.ln(4)

    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Principais Insights e Conclusões", ln=True)
    pdf.ln(2)
    
    pdf.set_font("helvetica", "", 10)
    _write_multiline(pdf, insights, line_height=6)

    pdf_data = pdf.output()

    # Streamlit download_button expects bytes for binary payloads.
    if isinstance(pdf_data, bytearray):
        return bytes(pdf_data)
    if isinstance(pdf_data, str):
        return pdf_data.encode("latin-1")
    return pdf_data