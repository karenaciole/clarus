"""Agente de engenharia do Clarus (LangChain `create_agent`).

O agente decide sozinho quais ferramentas chamar para responder:
- busca semântica nos documentos da sessão (RAG) ou no histórico;
- APIs REST de bibliotecas técnicas: PyPI (metadados de pacotes Python) e
  OSV.dev (vulnerabilidades conhecidas por versão).
"""
import re
from typing import Any

import requests
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool

from config.logging_config import app_logger
from src.engine import format_docs, to_lc_messages
from src.prompts import AGENT_SYSTEM_PROMPT

PYPI_URL = "https://pypi.org/pypi/{package}/json"
PYPI_VERSION_URL = "https://pypi.org/pypi/{package}/{version}/json"
OSV_QUERY_URL = "https://api.osv.dev/v1/query"
HTTP_TIMEOUT_SECONDS = 10
MAX_VULNS_LISTED = 10

# Nomes de pacote e versões válidos (PEP 508 / PEP 440); evita injeção na URL.
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9.+!_-]{1,64}$")


def _invalid_package(name: str) -> str | None:
    if not _PACKAGE_RE.fullmatch(name or ""):
        return f"Nome de pacote inválido: '{name}'."
    return None


def get_package_info(package_name: str, version: str | None = None, session: Any = requests) -> str:
    """Consulta a API JSON do PyPI e devolve um resumo do pacote."""
    if error := _invalid_package(package_name):
        return error
    if version and not _VERSION_RE.fullmatch(version):
        return f"Versão inválida: '{version}'."

    url = (
        PYPI_VERSION_URL.format(package=package_name, version=version)
        if version
        else PYPI_URL.format(package=package_name)
    )
    try:
        resp = session.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        return f"Erro ao consultar o PyPI: {e}"
    if resp.status_code == 404:
        return f"Pacote '{package_name}'{f' na versão {version}' if version else ''} não encontrado no PyPI."
    if resp.status_code != 200:
        return f"PyPI respondeu com status {resp.status_code}."

    info = resp.json().get("info", {})
    project_urls = info.get("project_urls") or {}
    homepage = info.get("home_page") or project_urls.get("Homepage") or project_urls.get("Source") or "-"
    return "\n".join([
        f"Fonte: PyPI ({url})",
        f"Pacote: {info.get('name', package_name)}",
        f"Versão: {info.get('version', '-')}{' (consultada)' if version else ' (mais recente)'}",
        f"Resumo: {info.get('summary') or '-'}",
        f"Python exigido: {info.get('requires_python') or 'não informado'}",
        f"Licença: {info.get('license_expression') or info.get('license') or 'não informada'}",
        f"Página: {homepage}",
    ])


def get_package_vulnerabilities(package_name: str, version: str, session: Any = requests) -> str:
    """Consulta a base OSV.dev por vulnerabilidades conhecidas de um pacote PyPI."""
    if error := _invalid_package(package_name):
        return error
    if not _VERSION_RE.fullmatch(version or ""):
        return f"Versão inválida: '{version}'."

    payload = {"package": {"name": package_name, "ecosystem": "PyPI"}, "version": version}
    try:
        resp = session.post(OSV_QUERY_URL, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        return f"Erro ao consultar o OSV.dev: {e}"
    if resp.status_code != 200:
        return f"OSV.dev respondeu com status {resp.status_code}."

    vulns = resp.json().get("vulns", [])
    if not vulns:
        return f"Fonte: OSV.dev. Nenhuma vulnerabilidade conhecida para {package_name}=={version}."

    lines = [f"Fonte: OSV.dev. {len(vulns)} vulnerabilidade(s) conhecida(s) para {package_name}=={version}:"]
    for v in vulns[:MAX_VULNS_LISTED]:
        aliases = ", ".join(a for a in v.get("aliases", []) if a.startswith("CVE-"))
        fixed = sorted({
            event["fixed"]
            for affected in v.get("affected", [])
            for rng in affected.get("ranges", [])
            for event in rng.get("events", [])
            if "fixed" in event
        })
        summary = v.get("summary") or (v.get("details") or "")[:160] or "sem descrição"
        lines.append(
            f"- {v.get('id')}{f' ({aliases})' if aliases else ''}: {summary}"
            f" | corrigido em: {', '.join(fixed) if fixed else 'sem versão corrigida informada'}"
        )
    if len(vulns) > MAX_VULNS_LISTED:
        lines.append(f"... e mais {len(vulns) - MAX_VULNS_LISTED}.")
    return "\n".join(lines)


def build_tools(rag, session: Any = requests) -> list[BaseTool]:
    """Ferramentas do agente. `rag` precisa expor `retrieve(query, historical=...)`."""

    @tool
    def search_documents(query: str) -> str:
        """Busca trechos relevantes nos documentos enviados na sessão atual."""
        docs = rag.retrieve(query, historical=False)
        return format_docs(docs) if docs else "Nenhum trecho relevante encontrado nos documentos da sessão."

    @tool
    def search_document_history(query: str) -> str:
        """Busca trechos em TODOS os documentos já indexados, inclusive de sessões anteriores.
        Use apenas quando o usuário pedir explicitamente documentos antigos ou históricos."""
        docs = rag.retrieve(query, historical=True)
        return format_docs(docs) if docs else "Nenhum trecho relevante encontrado no histórico."

    @tool
    def get_python_package_info(package_name: str, version: str = "") -> str:
        """Consulta o PyPI: versão mais recente (ou a versão informada), resumo, licença e
        versão do Python exigida de um pacote Python."""
        return get_package_info(package_name, version or None, session=session)

    @tool
    def check_package_vulnerabilities(package_name: str, version: str) -> str:
        """Consulta o OSV.dev para listar vulnerabilidades conhecidas (CVEs) de uma versão
        específica de um pacote Python e as versões que as corrigem."""
        return get_package_vulnerabilities(package_name, version, session=session)

    return [search_documents, search_document_history, get_python_package_info, check_package_vulnerabilities]


def build_agent(llm: BaseChatModel, rag, session: Any = requests):
    """Cria o agente com as ferramentas do Clarus."""
    return create_agent(model=llm, tools=build_tools(rag, session=session), system_prompt=AGENT_SYSTEM_PROMPT)


def run_agent(agent, question: str, chat_history: list[dict] | None = None, history_turns: int = 5) -> str:
    """Executa o agente e devolve o texto da resposta final."""
    messages = to_lc_messages(chat_history, history_turns, current_query=question)
    messages.append(HumanMessage(content=question))
    result = agent.invoke({"messages": messages})
    final = result["messages"][-1]
    app_logger.info(f"Agente respondeu após {len(result['messages']) - len(messages)} mensagem(ns) de raciocínio/ferramentas.")
    return final.text if hasattr(final, "text") else str(final.content)
