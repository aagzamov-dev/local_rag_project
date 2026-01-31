import os
import re
import logging
from openai import OpenAI

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

logging.getLogger("transformers").setLevel(logging.ERROR)

CHROMA_PATH = "./chroma"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

_chroma_client = None
_collection = None
_embedding_model = None
_llm = None


FALLBACK = "I don't have enough information to answer this."


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    llm_path = os.getenv("LLM_GGUF_PATH")
    if not llm_path:
        raise ValueError("Environment variable LLM_GGUF_PATH is not set.")

    _llm = Llama(
        model_path=llm_path,
        n_ctx=4096,
        n_gpu_layers=0,
        n_threads=max(1, os.cpu_count() or 8),
        n_batch=256,
        verbose=False,
    )
    return _llm


def query_documents(query, top_k=5):
    collection = get_collection()
    model = get_embedding_model()

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]
    distances = (
        results["distances"][0] if "distances" in results else [0.0] * len(documents)
    )

    retrieved = []
    for doc, meta, id_, dist in zip(documents, metadatas, ids, distances):
        retrieved.append(
            {"content": doc, "metadata": meta or {}, "id": id_, "distance": dist}
        )
    return retrieved


def _citation_for_item(item):
    src = item["metadata"].get("source", "unknown")
    filename = os.path.basename(src)
    chunk_id = item["metadata"].get("chunk_id")
    if chunk_id is None:
        chunk_id = item.get("id", "?")
    return f"[{filename}#{chunk_id}]"


def format_prompt(query, retrieved_items):
    context_blocks = []
    for item in retrieved_items:
        cite = _citation_for_item(item)
        context_blocks.append(f"{cite}\n{item['content']}")
    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are an efficient expert assistant.

Rules:
- Use ONLY facts that are explicitly stated verbatim in the provided information.
- Do NOT use general knowledge, assumptions, background knowledge, or external facts.
- Before answering, verify the provided information is directly and clearly relevant to the question.
  If relevance is unclear or missing, output exactly:
  {FALLBACK}
- If the question requires any detail that is not explicitly stated in the provided information, output exactly:
  {FALLBACK}
- If the fallback sentence is used, output it ALONE.
  Do NOT add explanations, reasoning, sources, or any additional text.
- Do NOT summarize, generalize, or conclude beyond the stated facts.
- Do NOT mention citations, filenames, chunk ids, sources, or the word "context".
- Do NOT include bracketed text of any kind.
- Write ONE concise paragraph (2–5 sentences), technical, factual, and neutral.


Context:
{context_str}

Question: {query}

Answer:"""
    return prompt


def run_openai_completion(prompt, api_key):
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep your answers concise and strictly based on the context provided entirely in the user prompt.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {str(e)}"


def finalize_answer(model_text, retrieved_items):
    text = model_text.strip()

    # Strip brackets first
    text = re.sub(r"\[[^\]]+\]", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    # Fix spacing before punctuation
    text = re.sub(r"\s+([,.!?])", r"\1", text)

    # Deterministic fallback check
    if FALLBACK.lower() in text.lower():
        return FALLBACK

    # Calculate percentages based on similarity
    # similarity = 1 - cosine_distance
    # Clip to [0, 1] just in case
    scores = []
    total_score = 0.0

    for item in retrieved_items:
        dist = item.get("distance", 1.0)
        # Handle cases where dist might be > 1 or < 0
        sim = max(0.0, 1.0 - dist)
        scores.append(sim)
        total_score += sim

    if total_score <= 0:
        total_score = 1.0  # Avoid division by zero

    sources_list = []
    seen = set()

    for item, score in zip(retrieved_items, scores):
        c = _citation_for_item(item)
        if c not in seen:
            seen.add(c)
            pct = (score / total_score) * 100
            sources_list.append(f"{c} ({int(pct)}%)")

    return f"{text}\n\nSources: {', '.join(sources_list)}"
