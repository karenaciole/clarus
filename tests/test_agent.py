from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from src.agent import build_agent, get_package_info, get_package_vulnerabilities, run_agent

from .conftest import ToolCallingFakeModel


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, get=None, post=None):
        self._get, self._post = get, post
        self.calls = []

    def get(self, url, timeout):
        self.calls.append(("GET", url, None))
        return self._get

    def post(self, url, json, timeout):
        self.calls.append(("POST", url, json))
        return self._post


class FakeRag:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, historical=False):
        self.calls.append((query, historical))
        return [Document(page_content="Dependência: requests==2.19.0", metadata={"file_name": "spec.pdf"})]


PYPI_PAYLOAD = {"info": {"name": "requests", "version": "2.32.3", "summary": "HTTP for Humans.",
                         "requires_python": ">=3.8", "license": "Apache-2.0",
                         "project_urls": {"Homepage": "https://requests.readthedocs.io"}}}

OSV_PAYLOAD = {"vulns": [{
    "id": "GHSA-x84v-xcm2-53pg",
    "aliases": ["CVE-2018-18074"],
    "summary": "Insufficiently Protected Credentials in Requests",
    "affected": [{"ranges": [{"events": [{"introduced": "0"}, {"fixed": "2.20.0"}]}]}],
}]}


def test_package_info_formats_pypi_response():
    session = FakeSession(get=FakeResponse(200, PYPI_PAYLOAD))
    out = get_package_info("requests", session=session)
    assert "Versão: 2.32.3 (mais recente)" in out
    assert "Licença: Apache-2.0" in out
    assert session.calls[0][1] == "https://pypi.org/pypi/requests/json"


def test_package_info_not_found_and_invalid_name():
    assert "não encontrado" in get_package_info("nao-existe", session=FakeSession(get=FakeResponse(404)))
    session = FakeSession(get=FakeResponse(200, PYPI_PAYLOAD))
    assert "inválido" in get_package_info("../../etc", session=session)
    assert session.calls == []


def test_vulnerabilities_lists_cve_and_fix():
    session = FakeSession(post=FakeResponse(200, OSV_PAYLOAD))
    out = get_package_vulnerabilities("requests", "2.19.0", session=session)
    assert "CVE-2018-18074" in out
    assert "corrigido em: 2.20.0" in out
    assert session.calls[0][2] == {"package": {"name": "requests", "ecosystem": "PyPI"}, "version": "2.19.0"}


def test_vulnerabilities_none_found():
    out = get_package_vulnerabilities("requests", "2.32.3", session=FakeSession(post=FakeResponse(200, {})))
    assert "Nenhuma vulnerabilidade" in out


def test_agent_chains_document_search_and_osv_tool():
    model = ToolCallingFakeModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "dependências"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "check_package_vulnerabilities",
                                           "args": {"package_name": "requests", "version": "2.19.0"}, "id": "c2"}]),
        AIMessage(content="O spec.pdf usa requests 2.19.0, afetado pela CVE-2018-18074. Atualize para 2.20.0+."),
    ])
    rag = FakeRag()
    session = FakeSession(post=FakeResponse(200, OSV_PAYLOAD))
    agent = build_agent(model, rag, session=session)

    answer = run_agent(agent, "As dependências do documento têm vulnerabilidades?")

    assert "CVE-2018-18074" in answer
    assert rag.calls == [("dependências", False)]
    assert session.calls[0][0] == "POST"
    tool_outputs = [m for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert any("CVE-2018-18074" in m.content for m in tool_outputs)
