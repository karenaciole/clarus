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
        self.set_fill_color(15, 23, 42) 
        self.rect(0, 0, 210, 25, "F")
        
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 14)
        self.set_y(8)
        self.cell(0, 10, "Relatório Executivo de Inteligência Artificial | Clarus", align="C")
        self.ln(20) 
        self.set_text_color(0, 0, 0) 

    def footer(self):
        self.set_y(-15)
        self.set_text_color(100, 116, 139) 
        self.set_font("helvetica", "I", 8)
        
        self.set_draw_color(203, 213, 225) 
        self.line(15, self.get_y(), 195, self.get_y())
        
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}} - Gerado automaticamente via AWS & Google Gemini", align="C")

def generate_technical_report(report_data, documents, filename="relatorio_tecnico.pdf"):
    md = MarkdownIt('commonmark', {'breaks': True, 'html': True})

    def _to_text(value):
        if value is None:
            return ""
        text = getattr(value, "response", str(value))
        return str(text)

    def _write_multiline(pdf_obj, text):
        html = md.render(_to_text(text))
        pdf_obj.write_html(html)

    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    pdf.ln(10)
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(13, 148, 136) 
    pdf.cell(0, 15, "Síntese e Análise de Documentos", align="C", ln=True)
    
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(71, 85, 105) 
    pdf.cell(0, 10, f"Data da Análise: {datetime.now().strftime('%d/%m/%Y às %H:%M')}", align="C", ln=True)
    pdf.ln(15)

    pdf.set_fill_color(241, 245, 249) 
    pdf.set_draw_color(203, 213, 225) 
    
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, " Escopo da Análise (Documentos Fonte):", border="TLR", fill=True, ln=True)
    
    pdf.set_font("helvetica", "", 11)
    for doc in documents:
        pdf.cell(0, 8, f"   - {doc}", border="LR", fill=True, ln=True)
    pdf.cell(0, 5, "", border="BLR", fill=True, ln=True) 
    
    pdf.ln(15)

    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 10, "1. Histórico da Conversa", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 11)
    _write_multiline(pdf, report_data["summary"])
    
    pdf.ln(10)
    
    # Linha divisória elegante
    pdf.set_draw_color(13, 148, 136)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(10)

    if pdf.get_y() > 220:
        pdf.add_page()
        
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 10, "2. Insights Técnicos e Recomendações", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 11)
    _write_multiline(pdf, report_data["insights"])

    pdf_data = pdf.output()

    if isinstance(pdf_data, bytearray):
        return bytes(pdf_data)
    if isinstance(pdf_data, str):
        return pdf_data.encode("latin-1")
    return pdf_data