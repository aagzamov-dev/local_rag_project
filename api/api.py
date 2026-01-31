import shutil
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from rag_core import (
    get_llm,
    query_documents,
    format_prompt,
    get_chroma_client,
    run_openai_completion,
    finalize_answer,
)

# Load env vars from current directory (api/)
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    masked = openai_key[:4] + "*" * (len(openai_key) - 8) + openai_key[-4:]
    print(f"Loaded OpenAI Key: {masked}")
else:
    print("Warning: No OpenAI Key loaded.")

# Fix relative model path if it exists
model_path = os.getenv("LLM_GGUF_PATH")
if model_path:
    # If path is relative, resolve it against api/ directory for robustness
    if not os.path.isabs(model_path):
        clean_path = model_path.lstrip(".\\/")
        abs_path = BASE_DIR / clean_path
        os.environ["LLM_GGUF_PATH"] = str(abs_path)
        print(f"Resolved Model Path: {abs_path}")

app = FastAPI(title="Local RAG API", version="1.0")

# Enable CORS for SPA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance to avoid reloading
llm = None
DATA_DIR = "./data"


class ChatRequest(BaseModel):
    query: str
    model: str = "local"


class JsonCreateRequest(BaseModel):
    filename: str
    data: dict


@app.on_event("startup")
def startup_event():
    global llm
    try:
        if os.getenv("LLM_GGUF_PATH"):
            llm = get_llm()
            print("LLM Loaded.")
        else:
            print("Warning: LLM_GGUF_PATH not set. Chat endpoint will fail.")
    except Exception as e:
        print(f"Error loading LLM: {e}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    retrieved = query_documents(request.query, top_k=5)
    prompt = format_prompt(request.query, retrieved)

    response_text = ""

    if request.model == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "INSERT_YOUR_KEY" in api_key:
            # Just a warning or strictly fail? User implies "take token", I'll assume they will update.
            # But if I put placeholder, it will fail. I'll let it try.
            if not api_key:
                raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

        response_text = run_openai_completion(prompt, api_key)

    else:
        if not llm:
            raise HTTPException(status_code=500, detail="LLM not initialized.")

        stream = llm.create_completion(
            prompt,
            max_tokens=512,
            stop=["User:", "Question:", "Answer:", "<|im_end|>"],
            stream=True,
            temperature=0.1,
            repeat_penalty=1.1,
        )

        for output in stream:
            response_text += output["choices"][0]["text"]

    final_response = finalize_answer(response_text, retrieved)

    # Streaming the final string so frontend logic remains compatible
    def iter_final():
        yield final_response

    return StreamingResponse(iter_final(), media_type="text/plain")


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    saved_files = []
    for file in files:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    return {
        "message": "Files uploaded successfully",
        "files": saved_files,
        "note": "Run ingest logic to index these files.",
    }


@app.post("/create-json")
async def create_json_file(request: JsonCreateRequest):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    filename = request.filename
    if not filename.endswith(".json"):
        filename += ".json"

    file_path = os.path.join(DATA_DIR, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(request.data, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "JSON file created successfully", "path": file_path}


# To run: uvicorn api:app --reload
