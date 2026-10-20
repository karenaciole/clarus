from src.loaders import load_documents


def test_loads_supported_files_with_file_name(tmp_path):
    (tmp_path / "notas.txt").write_text("Requisito: usar Python 3.11", encoding="utf-8")
    (tmp_path / "imagem.png").write_bytes(b"\x89PNG")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata["file_name"] == "notas.txt"
    assert "Python 3.11" in docs[0].page_content


def test_bad_file_does_not_break_batch(tmp_path):
    (tmp_path / "quebrado.pdf").write_bytes(b"nao e um pdf")
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.metadata["file_name"] for d in docs] == ["ok.txt"]
