import os
import glob
import json
import uuid
from pypdf import PdfReader
from rag_core import get_collection, get_embedding_model

DATA_DIR = "./data"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def load_documents():
    documents = []
    # Load PDF
    for path in glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True):
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append({
                        "content": text,
                        "metadata": {"source": os.path.basename(path), "page": i + 1, "type": "pdf"}
                    })
        except Exception as e:
            print(f"Error reading {path}: {e}")

    # Load TXT and MD
    for ext in ["**/*.txt", "**/*.md"]:
        for path in glob.glob(os.path.join(DATA_DIR, ext), recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text:
                        documents.append({
                            "content": text,
                            "metadata": {"source": os.path.basename(path), "type": ext.split(".")[-1]}
                        })
            except Exception as e:
                print(f"Error reading {path}: {e}")

    # Load JSON
    for path in glob.glob(os.path.join(DATA_DIR, "**/*.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, indent=2, ensure_ascii=False)
                if text:
                    documents.append({
                        "content": text,
                        "metadata": {"source": os.path.basename(path), "type": "json"}
                    })
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    return documents

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks

def ingest():
    print("Loading documents...")
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} source documents.")
    
    print("Chunking...")
    chunked_docs = []
    ids = []
    metadatas = []
    documents_content = []
    
    for doc in raw_docs:
        chunks = chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['metadata']['source']}#{i}"
            
            # Update metadata with chunk_id
            meta = doc["metadata"].copy()
            meta["chunk_id"] = i
            
            chunked_docs.append(chunk)
            ids.append(str(uuid.uuid4()))
            metadatas.append(meta)
            documents_content.append(chunk)

    if not chunked_docs:
        print("No documents to ingest.")
        return

    print("Generating embeddings and storing...")
    collection = get_collection()
    model = get_embedding_model()
    
    embeddings = model.encode(documents_content).tolist()
    
    # Batch extraction to avoid hitting limits if any (though local is fine usually)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            documents=documents_content[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Processed batch {i} to {end}")

    print("Ingestion complete.")

if __name__ == "__main__":
    # To run this script: python ingest.py
    ingest()
