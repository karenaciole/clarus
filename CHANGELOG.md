# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [0.1.0] - 2026-03-13 - Initial Version

This was the initial version of the project.

-   Basic Streamlit user interface for file upload and chat.
-   Core RAG implementation using LlamaIndex and Ollama.
-   All configurations were hardcoded within the source files.
-   The index was rebuilt every time the application started.
-   No conversational history was maintained between queries.
-   Basic PDF report generation with `reportlab`.
