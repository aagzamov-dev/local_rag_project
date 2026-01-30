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
from rag_core import get_llm, query_documents, format_prompt, get_chroma_client

# Load env vars from project root (one level up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

# Fix relative model path if it exists
model_path = os.getenv("LLM_GGUF_PATH")
if model_path and (model_path.startswith(".") or model_path.startswith("\\")):
    # Assuming relative path in .env is relative to Project Root, not api/ dir
    # Remove leading characters if needed or just join
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
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized.")
    
    retrieved = query_documents(request.query, top_k=3)
    prompt = format_prompt(request.query, retrieved)
    
    # We need to capture the full response to append citations properly
    # Streaming with appended content is tricky without client-side logic
    # switching to standard response for stability with the new strict format
    
    response_text = ""
    stream = llm.create_completion(
        prompt,
        max_tokens=512,
        stop=["User:", "Question:", "Answer:", "<|im_end|>"],
        stream=True,
        temperature=0.1,
        repeat_penalty=1.1
    )
    
    for output in stream:
        response_text += output['choices'][0]['text']
        
    from rag_core import finalize_answer
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
        
    return {"message": "Files uploaded successfully", "files": saved_files, "note": "Run ingest logic to index these files."}

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
