import os
import re
import logging
import json
from openai import OpenAI
from tools import get_openai_tools, get_local_tools_prompt, AVAILABLE_TOOLS

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


def run_openai_agent(query, context_items, api_key):
    """
    OpenAI Agent with Tool Calling support.
    Constructs a prompt that integrates RAG context but allows tools if needed.
    """
    client = OpenAI(api_key=api_key)
    tools = get_openai_tools()

    # Build context string
    context_blocks = []
    for item in context_items:
        cite = _citation_for_item(item)
        context_blocks.append(f"{cite}\n{item['content']}")
    context_str = "\n\n".join(context_blocks)

    system_prompt = """You are a helpful and intelligent assistant.
    
    You have access to a knowledge base (CONTEXT) and external tools.
    
    STRATEGY:
    1. First, check if the user's question can be answered using the provided CONTEXT. 
       - If YES, verify the facts, use them to answer, and cite the source filename in your answer if relevant.
    2. If the CONTEXT is missing, irrelevant, or does not contain the answer, you SHOULD use the available tools (e.g. web_search, get_weather) to find the answer.
    3. If the question is about current events, weather, or specific real-time data, IGNORE the context and immediate use the tools.
    
    Do not mention "I don't have enough information" unless you have tried both the context AND the tools and failed.
    """

    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    try:
        # First call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )

        response_msg = response.choices[0].message

        # Check if tool call
        if response_msg.tool_calls:
            # Append assistant's request to history
            messages.append(response_msg)

            # Execute tools
            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if func_name in AVAILABLE_TOOLS:
                    func_to_call = AVAILABLE_TOOLS[func_name]
                    try:
                        tool_output = func_to_call(**args)
                    except Exception as e:
                        tool_output = f"Error: {str(e)}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_output),
                        }
                    )

            # Second call to get final answer
            final_res = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, temperature=0.1
            )
            return final_res.choices[0].message.content, True

        else:
            return response_msg.content, False

    except Exception as e:
        return f"OpenAI Agent Error: {str(e)}", False


def run_local_agent(query, context_items, llm):
    """
    Local Agent using ReAct (Thought-Action-Observation) styling.
    """
    tool_instructions = get_local_tools_prompt()

    # Build text context
    context_blocks = []
    for item in context_items:
        cite = _citation_for_item(item)
        context_blocks.append(f"{cite}\n{item['content']}")
    context_str = "\n\n".join(context_blocks)

    # Prompt engineering for local model (Llama-3 style or Alpaca style)
    # We need a prompt that encourages checking context first, then tools.

    full_prompt = f"""You are a helpful assistant.

TOOLS:
{tool_instructions}

CONTEXT:
{context_str}

INSTRUCTIONS:
1. Try to answer the User's Question using the CONTEXT above.
2. If the CONTEXT is not relevant or missing the answer, use a tool (like web_search or get_weather).
3. To use a tool, output: req_tool: tool_name("arg")
4. Do NOT say "I don't have enough information" without trying a tool first.

User: {query}
Assistant:"""

    # 1. Generate (Stop at req_tool: or normal stops)
    output = llm.create_completion(
        full_prompt,
        max_tokens=256,
        stop=["Observation:", "User:", "<|im_end|>"],
        temperature=0.1,
    )
    text = output["choices"][0]["text"]

    if "req_tool:" in text:
        try:
            tool_request = text.split("req_tool:")[-1].strip()
            if "(" in tool_request and ")" in tool_request:
                func_name = tool_request.split("(")[0].strip()
                arg_raw = tool_request.split("(")[1].split(")")[0]
                arg_val = arg_raw.strip('"').strip("'")

                if func_name in AVAILABLE_TOOLS:
                    print(f"Local Agent Calling: {func_name} with {arg_val}")
                    result = AVAILABLE_TOOLS[func_name](arg_val)

                    # Feed back observation
                    new_prompt = (
                        f"{full_prompt}{text}\nObservation: {result}\nAssistant:"
                    )

                    final_out = llm.create_completion(
                        new_prompt,
                        max_tokens=256,
                        stop=["User:", "<|im_end|>"],
                        temperature=0.1,
                    )
                    return final_out["choices"][0]["text"], True

            return text + "\n[System: Could not parse tool request properly]", False

        except Exception as e:
            return f"{text} [Error processing tool request: {e}]", False

    return text, False


def finalize_answer(model_text, retrieved_items, tool_used=False):
    text = model_text.strip()
    text = re.sub(r"\[[^\]]+\]", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?])", r"\1", text)

    if FALLBACK.lower() in text.lower() and len(text) < 100:
        return FALLBACK

    # HIDE SOURCES IF TOOL USED
    if tool_used:
        return text

    scores = []
    total_score = 0.0
    for item in retrieved_items:
        dist = item.get("distance", 1.0)
        sim = max(0.0, 1.0 - dist)
        scores.append(sim)
        total_score += sim

    if total_score <= 0:
        total_score = 1.0

    sources_list = []
    seen = set()
    for item, score in zip(retrieved_items, scores):
        c = _citation_for_item(item)
        if c not in seen:
            seen.add(c)
            pct = (score / total_score) * 100
            sources_list.append(f"{c} ({int(pct)}%)")

    return f"{text}\n\nSources: {', '.join(sources_list)}"
