# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release migrates the RAG engine from LlamaIndex to LangChain, adds an engineering agent with REST API tools, and introduces CI. It also improves container readiness for AWS deployments and ECR publishing.

### Added

-   **LangChain Agent Mode**: New agent built with `langchain.agents.create_agent` that chooses between document search tools and REST APIs of technical libraries (PyPI package metadata and OSV.dev vulnerability lookups). Toggle it in the sidebar.
-   **Automated Tests**: Unit tests with fake LLMs/embeddings for the conversational chain, report synthesis, embeddings fallback, loaders, and agent tool-calling, plus integration tests against a real PostgreSQL + pgvector.
-   **CI Pipeline**: GitHub Actions workflow running ruff, pytest (Python 3.11 and 3.12), pgvector integration tests, and the Docker image build.
-   **Container Port Exposure**: Exposed Streamlit port 8501 in the Docker image to simplify ECS/ECR deployments.

### Changed

-   **LlamaIndex → LangChain**: The RAG engine now uses LangChain LCEL chains, `langchain-postgres` (`PGVector`), `RecursiveCharacterTextSplitter`, and LangChain provider integrations for Gemini and Bedrock (Converse API). The public `RAG` interface used by the UI is unchanged.
-   **Report Synthesis**: LlamaIndex `tree_summarize` replaced by an explicit map-reduce tree summarization that respects the model context window.
-   **Database Driver**: `psycopg2` replaced by `psycopg` 3; hash lookups now use parameterized SQL.
-   **Docker Image**: Removed the NLTK download that was only needed by LlamaIndex.
-   **Env Defaults for Container Runtime**: Added safe defaults and quote normalization for environment variables so the app can boot when `.env` is not bundled into the image (useful for AWS task-level env/secret injection).

### Migration notes

-   Vectors indexed by LlamaIndex are stored in a different table layout. After deploying, documents are re-indexed into the `langchain-postgres` collection named by `VECTOR_STORE_TABLE` on their next upload.

## [1.1.0] - 2026-03-31

This release improves final report quality and stabilizes PDF generation when processing larger conversations and document sets.

### Added

-   **Evidence-Grounded Report Prompt**: The report template now explicitly uses only retrieved document context and requires the assistant to state when evidence is insufficient.
-   **Report Query Transformation**: Added a step that generates focused retrieval queries from recent conversation history before synthesis.
-   **Robust Retrieval Parsing**: Added normalization and de-duplication logic for generated retrieval queries.

### Changed

-   **Report Retrieval Strategy**: Replaced single broad retrieval with multi-query evidence collection to increase relevance and factual grounding in conclusions.
-   **Workload Limits for Stability**: Report generation now limits retrieval breadth (queries, top-k, and evidence nodes) to reduce latency and avoid overload.
-   **History Scope for Reporting**: Report query generation now uses a bounded recent window of chat history to improve signal-to-noise ratio.

### Fixed

-   **Streamlit Async Loop Error**: Resolved event loop/runtime issues in report synthesis by running synthesis in synchronous mode.
-   **PDF Generation Timeout**: Reduced timeout risk by lowering report-generation workload and adding guarded fallbacks when query generation fails.
-   **Off-Context Conclusions**: Improved grounding so final insights and recommendations stay aligned with retrieved document evidence.

## [1.0.0] - 2026-03-31

This major release marks a significant refactoring and enhancement of the Clarus application, moving it from a basic prototype to a robust and configurable RAG system.

### Added

-   **Multi-Provider LLM Support**: The engine was refactored to support both local models via Ollama and cloud models via Google Gemini, selectable through an environment variable.
-   **Persistent Index Caching**: Implemented a system to save the `LlamaIndex` vector store to disk. This dramatically improves startup time, as the index is only rebuilt when the source documents change.
-   **File Fingerprinting**: A hashing mechanism (`_fingerprint_uploads`) was added to detect changes in uploaded files, ensuring the index cache is used correctly.
-   **Conversational Memory**: The chat engine now maintains a history of the conversation, allowing for more context-aware and follow-up questions.
-   **Centralized Logging System**: A new logging configuration (`config/logging_config.py`) was created to provide structured logging to both the console and a `clarus.log` file, capturing key events, configurations, and errors.
-   **Detailed PDF Report Generation**: Added the capability to generate a final PDF report summarizing the chat history, insights, risks, and recommendations.
-   **Environment-Based Configuration**: The application now uses a `.env` file for all major configurations, making it easier to manage settings without changing the code.

### Changed

-   **Prompt Engineering**: Separated the system prompts. A general-purpose prompt is used for the interactive chat, while a more detailed, structured prompt is used specifically for generating the final report.
-   **PDF Generation Library**: Replaced `reportlab` with `fpdf2` to provide better support for Unicode characters and Markdown rendering in the final PDF report.
-   **API Usage**: Updated the code to use the correct `.chat()` method for the LlamaIndex `CondensePlusContextChatEngine`.

### Fixed

-   **`ModuleNotFoundError`**: Corrected the Python path to ensure all project modules are discoverable at runtime.
-   **`AttributeError`**: Fixed errors related to incorrect method calls on the LlamaIndex chat engine.
-   **Data Type Mismatch**: Resolved an error by converting the chat history from a list of dictionaries to a list of `ChatMessage` objects as required by the LlamaIndex API.
-   **PDF Unicode Errors**: Solved character encoding issues in PDF reports by switching to the `fpdf2` library.
-   **PDF Markdown Rendering**: Fixed an issue where Markdown styles (like bold) were not appearing in the generated PDF.
-   **Context Window Overflow**: Fixed a `ValueError` during report generation by refactoring the query engine to use a `tree_summarize` response synthesizer, preventing the model's context limit from being exceeded.
-   **Prompt Template Type Error**: Resolved an `AttributeError` by ensuring the report prompt was wrapped in a LlamaIndex `PromptTemplate` object before being passed to the response synthesizer.

## [0.1.0] - 2026-03-13 - Initial Version

This was the initial version of the project.

-   Basic Streamlit user interface for file upload and chat.
-   Core RAG implementation using LlamaIndex and Ollama.
-   All configurations were hardcoded within the source files.
-   The index was rebuilt every time the application started.
-   No conversational history was maintained between queries.
-   Basic PDF report generation with `reportlab`.
