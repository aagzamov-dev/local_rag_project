# Local RAG Project

A clean Python 3.11 RAG system using ChromaDB, Llama-cpp-python, and Sentence Transformers.

## Prerequisites

1.  **Environment**: Ensure you are in the python 3.11 environment.
2.  **Model**: Download the model from [here](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf) and place it at `.\models\qwen2.5-1.5b-instruct-q4_k_m.gguf`.
3.  **Environment Variable**:
    You **MUST** set the model path before running the scripts.
    ```bash
    set LLM_GGUF_PATH=.\models\qwen2.5-1.5b-instruct-q4_k_m.gguf
    ```

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Ingest Data

Place your PDF, TXT, MD, or JSON files in the `./data` folder.
Run the ingestion script to create the database:

```bash
python ingest.py
```

*Expected Output*: "Ingestion complete." and creation of `./chroma` directory.

## 3. CLI Chat

Run the interactive chat interface:

```bash
python chat.py
```

## 4. API & Swagger UI

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Upload Endpoint**: `/upload` (POST)
- **Chat Endpoint**: `/chat` (POST)
- **JSON Creation**: `/create-json` (POST)

## 5. Clean Output
The project is configured to suppress most warnings (HuggingFace symlinks, transformers logs) for a clean console experience.
