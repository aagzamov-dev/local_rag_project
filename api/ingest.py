import re
import os
import glob
import json
import uuid
from pathlib import Path
from pypdf import PdfReader
from rag_core import get_collection, get_embedding_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))


def load_documents():
    documents = []
    # Load PDF
    for path in glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True):
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(
                        {
                            "content": text,
                            "metadata": {
                                "source": os.path.basename(path),
                                "page": i + 1,
                                "type": "pdf",
                            },
                        }
                    )
        except Exception as e:
            print(f"Error reading {path}: {e}")

    # Load TXT and MD
    for ext in ["**/*.txt", "**/*.md"]:
        for path in glob.glob(os.path.join(DATA_DIR, ext), recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text:
                        documents.append(
                            {
                                "content": text,
                                "metadata": {
                                    "source": os.path.basename(path),
                                    "type": ext.split(".")[-1],
                                },
                            }
                        )
            except Exception as e:
                print(f"Error reading {path}: {e}")

    # Load JSON
    for path in glob.glob(os.path.join(DATA_DIR, "**/*.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, indent=2, ensure_ascii=False)
                if text:
                    documents.append(
                        {
                            "content": text,
                            "metadata": {
                                "source": os.path.basename(path),
                                "type": "json",
                            },
                        }
                    )
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return documents


def recursive_split(text, chunk_size, chunk_overlap):
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(text, separators):
        final_chunks = []
        if not separators:
            return [text]

        sep = separators[0]
        new_separators = separators[1:]

        if sep == "":
            # separate by characters as last resort
            splits = list(text)
        else:
            splits = text.split(sep)

        current_chunk = []
        current_length = 0

        for split in splits:
            if sep != "":
                split_len = len(split) + len(sep)
            else:
                split_len = len(split)

            if current_length + split_len > chunk_size:
                if current_length > 0:
                    joined = (
                        sep.join(current_chunk) if sep != "" else "".join(current_chunk)
                    )
                    final_chunks.append(joined)
                    current_chunk = []
                    current_length = 0

                # If the single split is simplified too big, recurse
                if split_len > chunk_size:
                    sub_chunks = _split_text(split, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    current_chunk.append(split)
                    current_length += split_len
            else:
                current_chunk.append(split)
                current_length += split_len

        if current_chunk:
            joined = sep.join(current_chunk) if sep != "" else "".join(current_chunk)
            final_chunks.append(joined)

        return final_chunks

    # Initial split
    naive_chunks = _split_text(text, separators)

    # Merge with overlap
    merged_chunks = []
    if not naive_chunks:
        return []

    current_doc = naive_chunks[0]

    for i in range(1, len(naive_chunks)):
        next_doc = naive_chunks[i]

        if len(current_doc) + len(next_doc) < chunk_size:
            current_doc += separators[1] + next_doc  # heuristic join
        else:
            merged_chunks.append(current_doc)
            # Create overlap
            overlap_len = max(0, len(current_doc) - chunk_overlap)
            current_doc = current_doc[overlap_len:] + separators[1] + next_doc

    merged_chunks.append(current_doc)

    return merged_chunks


def chunk_text(text, chunk_size, overlap):
    # wrapper to clean text first
    clean_text = re.sub(r"\s+", " ", text).strip()
    return recursive_split(clean_text, chunk_size, overlap)


def ingest():
    print("Loading documents...")
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} source documents.")

    print("Chunking with Recursive splitting...")
    chunked_docs = []
    ids = []
    metadatas = []
    documents_content = []

    for doc in raw_docs:
        chunks = chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            # Update metadata
            meta = doc["metadata"].copy()
            meta["chunk_id"] = i
            # Add length metadata for filtering if needed
            meta["length"] = len(chunk)

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

    # IMPORTANT: E5 models need 'passage: ' prefix for documents
    docs_for_embedding = [f"passage: {d}" for d in documents_content]

    embeddings = model.encode(docs_for_embedding, normalize_embeddings=True).tolist()

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            documents=documents_content[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"Processed batch {i} to {end}")

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
