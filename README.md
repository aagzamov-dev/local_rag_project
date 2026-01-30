# Local RAG Project

A powerful, privacy-focused Retrieval-Augmented Generation (RAG) system capable of running entirely locally. This project combines a robust Python backend for document processing and LLM inference with a modern React-based frontend.

## 🌟 Features

-   **Privacy First**: Runs completely offline using local models.
-   **RAG Architecture**: Ingests your documents (PDF, TXT, MD, JSON) for context-aware answers.
-   **Tech Stack**:
    -   **Backend**: Python 3.11, FastAPI, ChromaDB (Vector Store), Llama-cpp-python (Local Inference), Sentence Transformers (Embeddings).
    -   **Frontend**: React, TypeScript, Vite.
-   **Model**: Optimised for `Qwen2.5-1.5b-instruct-q4_k_m.gguf`.

## 📂 Project Structure

-   `api/`: The backend server and ingestion scripts.
-   `spa/`: The frontend Single Page Application.

## 🚀 Quick Start

### 1. Backend Setup (`api/`)

Please refer to the [API README](./api/README.md) for detailed instructions.

**Summary:**
1.  Navigate to `api/`.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Download the model: [Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf) and place it in `api/models/`.
4.  Set the environment variable `LLM_GGUF_PATH`.
5.  Ingest data: `python ingest.py`.
6.  Start the server: `uvicorn api:app --reload`.

### 2. Frontend Setup (`spa/`)

1.  Navigate to `spa/`:
    ```bash
    cd spa
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```

## 📄 License
[MIT](LICENSE)
