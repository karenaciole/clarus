# Clarus Technical Document Analysis Assistant

Clarus is a RAG (Retrieval-Augmented Generation) system designed to assist in the analysis of technical documents. It uses a language model to extract key information such as requirements, risks, and recommendations from a given set of documents.

## Features

-   **Conversational Interface**: Chat with an AI assistant to analyze your documents.
-   **Multi-Document Support**: Upload multiple documents in PDF, DOCX, or TXT format.
-   **Configurable RAG Pipeline**: Tune the RAG system's parameters for optimal performance.
-   **Dual LLM Provider Support**: Switch between local models with Ollama and cloud models with Google Gemini.
-   **Persistent Caching**: Document indexes are cached to speed up processing.
-   **PDF Report Generation**: Summarize the analysis in a downloadable PDF report.

## Setup and Configuration

### 1. Prerequisites

-   Python 3.9+
-   Pip (Python package installer)

### 2. Installation

Clone the repository and install the required Python packages:

```bash
git clone <repository-url>
cd clarus
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root of the project by copying the example file:

```bash
cp .env_example .env
```

Open the `.env` file and configure the variables as needed. Below is a description of the available settings:

| Variable                  | Description                                                                          | Default                  |
| ------------------------- | ------------------------------------------------------------------------------------ | ------------------------ |
| `LLM_PROVIDER`            | The language model provider to use. Can be `ollama` or `gemini`.                     | `ollama`                 |
| `OLLAMA_MODEL_NAME`       | The name of the model to use with Ollama (e.g., `mistral`, `llama2`).                | `mistral`                |
| `EMBEDDING_MODEL_NAME`    | The sentence-transformer model to use for embeddings.                                | `BAAI/bge-m3`            |
| `CHUNK_SIZE`              | The size of text chunks (in tokens) for document indexing.                           | `512`                    |
| `CHUNK_OVERLAP`           | The number of tokens to overlap between adjacent chunks.                             | `128`                    |
| `SIMILARITY_TOP_K`        | The number of most similar chunks to retrieve from the vector store.                 | `5`                      |
| `SIMILARITY_CUTOFF`       | The minimum similarity score (0.0 to 1.0) for a chunk to be considered.              | `0.0`                    |
| `HISTORY_TURNS`           | The number of past conversation turns to include as context for the LLM.             | `5`                      |
| `PERSIST_INDEX`           | Whether to cache the document index on disk (`True` or `False`).                     | `True`                   |
| `DEFAULT_TEMPERATURE`     | The creativity/randomness of the LLM's responses (0.0 to 1.0).                       | `0.1`                    |
| `DEFAULT_REQUEST_TIMEOUT` | The timeout in seconds for requests to the LLM.                                      | `120.0`                  |

## How to Run

Once you have configured your `.env` file, you can run the Streamlit application:

```bash
streamlit run ui/app.py
```

The application will open in your web browser. You can then upload your documents, process them, and start asking questions.
