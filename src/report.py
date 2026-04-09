from fpdf import FPDF
from datetime import datetime
import sys
from pathlib import Path
from markdown_it import MarkdownIt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class Report(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 12)
        self.cell(0, 10, "Relatório de Análise Técnica feito pelo Assistente Inteligente", border=False, ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", align="C")

def generate_technical_report(report_data, documents, filename="relatorio_tecnico.pdf"):
    md = MarkdownIt()

    def _to_text(value):
        if value is None:
            return ""
        text = getattr(value, "response", str(value))
        try:
            return text.encode('latin-1', 'replace').decode('latin-1')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text.encode('utf-8', 'replace').decode('utf-8')

    def _write_multiline(pdf_obj, text):
        html = md.render(_to_text(text))
        pdf_obj.write_html(html)

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
    pdf.cell(0, 10, "Resumo da Conversa", ln=True)
    pdf.ln(2)
    
    pdf.set_font("helvetica", "", 10)
    _write_multiline(pdf, report_data["summary"])
    pdf.ln(4)

    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Principais Insights e Conclusões", ln=True)
    pdf.ln(2)
    
    pdf.set_font("helvetica", "", 10)
    _write_multiline(pdf, report_data["insights"])

    pdf_data = pdf.output()

    if isinstance(pdf_data, bytearray):
        return bytes(pdf_data)
    if isinstance(pdf_data, str):
        return pdf_data.encode("latin-1")
    return pdf_data