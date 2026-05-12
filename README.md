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
| `VECTOR_STORE_TABLE` | Nome da tabela interna de indexação vetorial. | `clarus_vectors` |

*(Consulte `config/settings.py` para ver todos os parâmetros suportados)*.

### 4. Rodando o App
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
