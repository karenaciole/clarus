"""Carregamento de documentos como `Document` do LangChain."""
from pathlib import Path
from typing import Callable

import docx2txt
from langchain_core.documents import Document
from pypdf import PdfReader

from config.logging_config import app_logger


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    return [
        Document(page_content=text, metadata={"page": number})
        for number, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text() or "").strip()
    ]


def _load_docx(path: Path) -> list[Document]:
    return [Document(page_content=docx2txt.process(str(path)) or "")]


def _load_txt(path: Path) -> list[Document]:
    raw = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return [Document(page_content=raw.decode(encoding))]
        except UnicodeDecodeError:
            continue
    return []


_LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_txt,
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS)


def load_documents(directory: str | Path) -> list[Document]:
    """Lê os arquivos suportados de um diretório.

    Cada `Document` recebe `file_name` nos metadados, usado para associar o
    hash do arquivo (isolamento de contexto) e citar a fonte nas respostas.
    """
    documents: list[Document] = []
    for path in sorted(Path(directory).iterdir()):
        loader = _LOADERS.get(path.suffix.lower())
        if not path.is_file() or loader is None:
            continue
        try:
            loaded = loader(path)
        except Exception as e:  # noqa: BLE001 - um arquivo ruim não deve derrubar o lote
            app_logger.error(f"Não foi possível ler '{path.name}': {e}")
            continue
        for doc in loaded:
            doc.metadata["file_name"] = path.name
        documents.extend(loaded)
    return documents
