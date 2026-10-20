# Clarus: Assistente de Análise Documental RAG na AWS

Clarus é um sistema RAG (Retrieval-Augmented Generation) projetado para analisar e extrair informações de documentos técnicos. Construído com resiliência de nuvem (Cloud Native) em mente, ele usa as melhores práticas de Infraestrutura como Código (IaC) via AWS CDK e suporta _models foundations_ da AWS (Bedrock) e Google (Gemini).

## 🌟 Principais Funcionalidades

- **RAG com Isolamento de Contexto:** Previne alucinações restringindo a IA estritamente aos documentos analisados na sessão atual, utilizando hashes de arquivo.
- **Busca Histórica Inteligente (Query Classification):** A IA classifica a intenção do usuário e pode "abrir o escopo" para consultar o banco de dados histórico quando solicitado explicitamente.
- **Provedores de LLM Flexíveis:** Alternância fácil entre a API do Google Gemini e modelos do AWS Bedrock.
- **Relatórios Automatizados:** Sintetiza todo o histórico de chat e gera um relatório final detalhado em formato PDF.
- **Deploy Cloud-Native:** Infraestrutura arquitetada para a AWS usando Amazon EC2, Amazon RDS (PostgreSQL com `pgvector`), S3, AWS Secrets Manager e IAM Roles.

## 🏗️ Arquitetura AWS

A infraestrutura é provisionada inteiramente via **AWS CDK** (presente no diretório `infra/`).
- **Computação:** Interface web construída com Streamlit e empacotada via Docker, hospedada em uma instância EC2 (Amazon Linux 2023).
- **Banco de Dados Vetorial:** Amazon RDS executando PostgreSQL (versão 16) com a extensão nativa `pgvector`, rodando isolado em uma subnet privada.
- **Armazenamento:** Documentos processados vão para o Amazon S3 de forma segura através de VPC Gateway Endpoints.
- **FinOps (Controle de Custo Automático):** Alarme no Amazon CloudWatch monitora o uso de CPU e envia um sinal de `STOP` para a instância EC2 caso ela fique ociosa, evitando custos.
- **Segurança Nativa:** Nenhuma chave de API da AWS fica no código. A aplicação EC2 usa um *Instance Profile* (IAM Role) restrito, conectando-se ao Bedrock, S3 e Secrets Manager apenas com permissões necessárias de leitura. 

## 🧠 Implementação do RAG

O sistema RAG é orquestrado com **LangChain** (cadeias LCEL) e adota boas práticas de engenharia de software aplicadas à IA:

- **Provedores intercambiáveis:** `ChatGoogleGenerativeAI`/`GoogleGenerativeAIEmbeddings` (Gemini) ou `ChatBedrockConverse`/`BedrockEmbeddings` (AWS Bedrock), escolhidos por `LLM_PROVIDER` em `src/providers.py`. O resto do código depende apenas das interfaces do LangChain.
- **Ingestão Inteligente (Deduplicação):** Cada upload recebe um hash SHA-256. Antes de chamar a API de embeddings, o sistema consulta no `pgvector` quais hashes já foram indexados, evitando custo de tokens duplicado e "poluição" na base.
- **Chunking e Resiliência de Vetorização:** `RecursiveCharacterTextSplitter` com sobreposição; o wrapper `ResilientEmbeddings` aplica *delay* (Bedrock) e *fallback* 1-a-1 quando um lote é rejeitado por rate limit.
- **Contexto Isolado via Metadados:** O hash do arquivo vai para os metadados do `PGVector` (`langchain-postgres`). No chat, o retriever usa o filtro `{"file_hash": {"$in": [...]}}` para responder apenas com os arquivos da sessão atual.
- **Busca Histórica Inteligente:** Uma classificação da pergunta decide se o filtro é removido para consultar documentos antigos.
- **Conversa com memória (condense + context):** A pergunta é reescrita como pergunta independente a partir do histórico, os chunks são recuperados e o LLM responde citando o arquivo de origem.
- **Síntese Estruturada (map-reduce em árvore):** No relatório, o Clarus resume a conversa, gera sub-queries, recupera trechos por documento e os sintetiza em níveis, respeitando a janela de contexto do modelo.

## 🛠️ Modo Agente (LangChain `create_agent`)

Ative **Modo agente** na barra lateral para que um agente LangChain decida quais ferramentas usar:

| Ferramenta | O que faz | Fonte |
| --- | --- | --- |
| `search_documents` | Busca semântica nos documentos da sessão | pgvector |
| `search_document_history` | Busca em todos os documentos já indexados | pgvector |
| `get_python_package_info` | Versão mais recente, licença e Python exigido de um pacote | API REST do PyPI |
| `check_package_vulnerabilities` | Vulnerabilidades conhecidas (CVEs) de uma versão e versões corrigidas | API REST do OSV.dev |

Exemplo: *"Quais dependências esta especificação usa e alguma tem vulnerabilidade conhecida?"* → o agente busca as dependências no documento e consulta o OSV.dev para cada uma.

## 🚀 Como Executar Localmente

### 1. Pré-requisitos
- Python 3.11+
- Instância PostgreSQL rodando com a extensão `pgvector` instalada.
- Docker (opcional, para construir a imagem da aplicação).

### 2. Instalação

```bash
git clone <repository-url>
cd clarus
pip install -r requirements.txt
```

### 3. Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
cp .env_example .env
```

| Variável | Descrição | Exemplo |
| --- | --- | --- |
| `LLM_PROVIDER` | Provedor de IA a ser usado (`bedrock` ou `gemini`). | `gemini` |
| `GOOGLE_API_KEY` | (Apenas se usar Gemini) Chave da API do Google. | `AIzaSy...` |
| `AWS_REGION` | Região da AWS para chamadas do Bedrock e do banco. | `us-east-1` |
| `DB_HOST` | Host do banco RDS PostgreSQL. | `localhost` |
| `DB_USER` | Usuário do banco de dados. | `clarus_admin` |
| `DB_PASSWORD` | Senha de acesso do usuário de DB. | `***` |
| `VECTOR_STORE_TABLE` | Nome da coleção vetorial no `langchain-postgres`. | `clarus_vectors` |

*(Consulte `config/settings.py` para ver todos os parâmetros suportados)*.

### 4. Testes

```bash
pip install -r requirements-dev.txt
ruff check src tests
pytest
```

Os testes de integração com PostgreSQL + pgvector rodam quando `CLARUS_TEST_DB_HOST` está definido (no CI, um container `pgvector/pgvector:pg16` é usado). O pipeline do GitHub Actions (`.github/workflows/ci.yml`) roda lint, testes unitários, testes de integração e o build da imagem Docker.

### 5. Rodando o App
```bash
streamlit run ui/app.py
```
Acesse `http://localhost:8501` em seu navegador para começar a analisar os documentos.

## ☁️ Deploy via AWS CDK

Para provisionar a infraestrutura em sua conta AWS:
```bash
cd infra/
cdk synth
cdk deploy
```
*Atenção: Garanta que você configurou as credenciais da AWS (`aws configure`) e que fez o push prévio da imagem Docker do app para um repositório ECR na mesma região.*
