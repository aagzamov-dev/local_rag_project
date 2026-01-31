import os
import re
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from tools import get_openai_tools, get_local_tools_prompt, AVAILABLE_TOOLS

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

logging.getLogger("transformers").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma")
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.2"))
MIN_TOOL_SIMILARITY = float(os.getenv("RAG_MIN_TOOL_SIMILARITY", "0.35"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

_chroma_client = None
_collection = None
_embedding_model = None
_llm = None


FALLBACK = "I don't have enough information to answer this."

_CITATION_PATTERN = re.compile(r"\[[^\[\]]+#[^\[\]]+\]")

_RAG_ONLY_RULES = f"""You are a retrieval-augmented assistant.
Rules:
- Use ONLY the information in CONTEXT to answer.
- If the answer is not explicitly stated in CONTEXT, reply exactly: {FALLBACK}
- When answering from CONTEXT, cite sources inline as [filename#chunk_id].
- Do NOT invent citations or sources.
"""

_TIME_SENSITIVE_KEYWORDS = [
    "today",
    "now",
    "current",
    "currently",
    "latest",
    "recent",
    "news",
    "breaking",
    "price",
    "stock",
    "exchange rate",
    "rate",
    "forecast",
    "weather",
    "temperature",
    "humidity",
    "wind",
    "rain",
    "snow",
    "time",
]

_WEATHER_KEYWORDS = [
    "weather",
    "forecast",
    "temperature",
    "humidity",
    "wind",
    "rain",
    "snow",
]


def _is_time_sensitive(query: str) -> bool:
    q = (query or "").lower()
    return any(keyword in q for keyword in _TIME_SENSITIVE_KEYWORDS)


def _is_weather_query(query: str) -> bool:
    q = (query or "").lower()
    return any(keyword in q for keyword in _WEATHER_KEYWORDS)


def _extract_weather_location(query: str) -> Optional[str]:
    if not query:
        return None
    match = re.search(r"\b(?:in|at|for|on)\s+([A-Za-z0-9\s,.-]+)", query, re.IGNORECASE)
    if match:
        location = match.group(1).strip()
        location = re.sub(
            r"\b(today|now|right now|currently|this week|tonight)\b",
            "",
            location,
            flags=re.IGNORECASE,
        ).strip(" ,.-")
        return location or None
    match = re.search(r"\bweather\s+([A-Za-z0-9\s,.-]+)", query, re.IGNORECASE)
    if match:
        location = match.group(1).strip(" ,.-")
        return location or None
    return None


def _should_use_tool(query: str, context_items: List[Dict[str, Any]]) -> bool:
    if not context_items:
        return True
    max_sim = max((item.get("similarity", 0.0) for item in context_items), default=0.0)
    if max_sim < MIN_TOOL_SIMILARITY:
        return True
    return _is_time_sensitive(query)


def _should_use_context_only(query: str, context_items: List[Dict[str, Any]]) -> bool:
    if not context_items:
        return False
    max_sim = max((item.get("similarity", 0.0) for item in context_items), default=0.0)
    return max_sim >= MIN_TOOL_SIMILARITY and not _is_time_sensitive(query)


def _infer_tool_call(query: str) -> Tuple[str, Dict[str, Any]]:
    if _is_weather_query(query):
        location = _extract_weather_location(query) or query
        return "get_weather", {"location": location}
    return "web_search", {"query": query}


def _is_fallback_text(text: str) -> bool:
    if not text:
        return True
    return FALLBACK.lower() in text.lower()

_OPENAI_TOOL_RULES = f"""You are a retrieval-augmented assistant with tools.
Decision policy:
- If CONTEXT clearly contains the answer, use it and cite sources inline as [filename#chunk_id].
- If CONTEXT is missing or insufficient, use a tool.
- For time-sensitive or real-time questions, use a tool even if CONTEXT exists.
- If tools fail to produce an answer, reply exactly: {FALLBACK}
"""

_LOCAL_TOOL_RULES = f"""You are a retrieval-augmented assistant with tools.
Decision policy:
- If CONTEXT clearly contains the answer, use it and cite sources inline as [filename#chunk_id].
- If CONTEXT is missing or insufficient, use a tool.
- For time-sensitive or real-time questions, use a tool even if CONTEXT exists.
- If tools fail to produce an answer, reply exactly: {FALLBACK}

Tool call format (output exactly, no extra text):
TOOL_CALL {{"name":"tool_name","arguments":{{...}}}}
"""


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


def query_documents(query, top_k: Optional[int] = None, min_similarity: Optional[float] = None):
    query = (query or "").strip()
    if not query:
        return []

    if top_k is None:
        top_k = DEFAULT_TOP_K
    if min_similarity is None:
        min_similarity = MIN_SIMILARITY

    collection = get_collection()
    model = get_embedding_model()

    # IMPORTANT: E5 models need 'query: ' prefix for queries
    query_embedding = model.encode([f"query: {query}"], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0] if results else []

    retrieved = []
    for doc, meta, id_, dist in zip(documents, metadatas, ids, distances):
        sim = max(0.0, 1.0 - dist)
        retrieved.append(
            {
                "content": doc,
                "metadata": meta or {},
                "id": id_,
                "distance": dist,
                "similarity": sim,
            }
        )
    if min_similarity is not None:
        retrieved = [item for item in retrieved if item["similarity"] >= min_similarity]
    return retrieved


def _citation_for_item(item):
    src = item["metadata"].get("source", "unknown")
    filename = os.path.basename(src)
    chunk_id = item["metadata"].get("chunk_id")
    if chunk_id is None:
        chunk_id = item.get("id", "?")
    return f"[{filename}#{chunk_id}]"


def build_context(retrieved_items: List[Dict[str, Any]], max_chars: Optional[int] = None):
    if max_chars is None:
        max_chars = MAX_CONTEXT_CHARS

    blocks = []
    total = 0
    seen = set()
    for item in retrieved_items:
        cite = _citation_for_item(item)
        if cite in seen:
            continue
        seen.add(cite)
        block = f"{cite}\n{item['content']}".strip()
        if max_chars and total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block) + 2
    return "\n\n".join(blocks)


def format_prompt(query, retrieved_items):
    context_str = build_context(retrieved_items)
    if not context_str:
        context_str = "NONE"

    prompt = f"""{_RAG_ONLY_RULES}

CONTEXT:
{context_str}

QUESTION: {query}

ANSWER:"""
    return prompt


def _rag_only_messages(query: str, retrieved_items: List[Dict[str, Any]]):
    context_str = build_context(retrieved_items)
    if not context_str:
        context_str = "NONE"
    system_prompt = _RAG_ONLY_RULES
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]


def run_openai_agent(query, context_items, api_key):
    """
    OpenAI Agent with Tool Calling support.
    Constructs a prompt that integrates RAG context but allows tools if needed.
    """
    client = OpenAI(api_key=api_key)
    tools = get_openai_tools()
    tool_trace = []

    if _should_use_context_only(query, context_items):
        messages = _rag_only_messages(query, context_items)
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""
            if _is_fallback_text(content):
                func_name, args = _infer_tool_call(query)
                try:
                    tool_output = AVAILABLE_TOOLS[func_name](**args)
                except Exception as e:
                    tool_output = f"Error: {str(e)}"
                tool_trace.append(
                    {"name": func_name, "arguments": args, "result": tool_output}
                )
                followup = [
                    *messages,
                    {
                        "role": "user",
                        "content": f"TOOL_RESULT:\n{tool_output}\n\nAnswer the question now.",
                    },
                ]
                final_res = client.chat.completions.create(
                    model=OPENAI_MODEL, messages=followup, temperature=0.1
                )
                final_text = final_res.choices[0].message.content or ""
                if _is_fallback_text(final_text):
                    return str(tool_output), True, tool_trace
                return final_text, True, tool_trace
            return content, False, tool_trace
        except Exception as e:
            return f"OpenAI Agent Error: {str(e)}", False, tool_trace

    context_str = build_context(context_items)
    if not context_str:
        context_str = "NONE"

    system_prompt = _OPENAI_TOOL_RULES
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    try:
        # First call
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )

        response_msg = response.choices[0].message

        # Check if tool call
        if response_msg.tool_calls:
            messages.append(response_msg)

            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                # Only strictly allowed tools
                if func_name not in AVAILABLE_TOOLS:
                    continue

                args = json.loads(tool_call.function.arguments)
                tool_output = f"Error: Tool {func_name} failed"

                try:
                    func_to_call = AVAILABLE_TOOLS[func_name]
                    tool_output = func_to_call(**args)
                except Exception as e:
                    tool_output = f"Error: {str(e)}"

                tool_trace.append(
                    {"name": func_name, "arguments": args, "result": tool_output}
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_output),
                    }
                )

            # Second call to get final answer
            final_res = client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, temperature=0.1
            )
            final_text = final_res.choices[0].message.content or ""
            if _is_fallback_text(final_text) and tool_trace:
                return str(tool_trace[-1].get("result", "")), True, tool_trace
            return final_text, True, tool_trace

        else:
            if _should_use_tool(query, context_items):
                func_name, args = _infer_tool_call(query)
                try:
                    tool_output = AVAILABLE_TOOLS[func_name](**args)
                except Exception as e:
                    tool_output = f"Error: {str(e)}"
                tool_trace.append(
                    {"name": func_name, "arguments": args, "result": tool_output}
                )
                followup = [
                    *messages,
                    {
                        "role": "user",
                        "content": f"TOOL_RESULT:\n{tool_output}\n\nAnswer the question now.",
                    },
                ]
                final_res = client.chat.completions.create(
                    model=OPENAI_MODEL, messages=followup, temperature=0.1
                )
                final_text = final_res.choices[0].message.content or ""
                if _is_fallback_text(final_text):
                    return str(tool_output), True, tool_trace
                return final_text, True, tool_trace

            return response_msg.content, False, tool_trace

    except Exception as e:
        return f"OpenAI Agent Error: {str(e)}", False, tool_trace


def run_local_agent(query, context_items, llm):
    """
    Local Agent using ReAct (Thought-Action-Observation) styling.
    """
    tool_trace = []

    if _should_use_context_only(query, context_items):
        messages = _rag_only_messages(query, context_items)
        text = _local_chat(llm, messages, max_tokens=512, temperature=0.1)
        if _is_fallback_text(text):
            func_name, args = _infer_tool_call(query)
            try:
                result = AVAILABLE_TOOLS[func_name](**args)
            except Exception as e:
                result = f"Tool error: {e}"
            tool_trace.append({"name": func_name, "arguments": args, "result": result})
            followup_messages = [
                *messages,
                {"role": "assistant", "content": "TOOL_CALL"},
                {"role": "user", "content": f"TOOL_RESULT:\n{result}\n\nAnswer the question now."},
            ]
            final_text = _local_chat(
                llm, followup_messages, max_tokens=512, temperature=0.1
            )
            if _is_fallback_text(final_text):
                return str(result), True, tool_trace
            return final_text, True, tool_trace
        return text, False, tool_trace

    tool_instructions = get_local_tools_prompt()
    context_str = build_context(context_items)
    if not context_str:
        context_str = "NONE"

    system_prompt = f"""{_LOCAL_TOOL_RULES}

Available tools:
{tool_instructions.strip()}
"""
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    text = _local_chat(llm, messages, max_tokens=512, temperature=0.1)
    tool_call = _parse_tool_call(text)

    if tool_call:
        func_name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        if func_name not in AVAILABLE_TOOLS:
            return FALLBACK, False, tool_trace

        try:
            args = _normalize_tool_args(func_name, args)
            result = AVAILABLE_TOOLS[func_name](**args)
        except Exception as e:
            result = f"Tool error: {e}"

        tool_trace.append({"name": func_name, "arguments": args, "result": result})

        followup_messages = [
            *messages,
            {"role": "assistant", "content": tool_call.get("raw", "TOOL_CALL")},
            {"role": "user", "content": f"TOOL_RESULT:\n{result}\n\nAnswer the question now."},
        ]
        final_text = _local_chat(llm, followup_messages, max_tokens=512, temperature=0.1)
        if _is_fallback_text(final_text):
            return str(result), True, tool_trace
        return final_text, True, tool_trace

    if _should_use_tool(query, context_items):
        func_name, args = _infer_tool_call(query)
        try:
            result = AVAILABLE_TOOLS[func_name](**args)
        except Exception as e:
            result = f"Tool error: {e}"
        tool_trace.append({"name": func_name, "arguments": args, "result": result})

        followup_messages = [
            *messages,
            {"role": "assistant", "content": "TOOL_CALL"},
            {"role": "user", "content": f"TOOL_RESULT:\n{result}\n\nAnswer the question now."},
        ]
        final_text = _local_chat(llm, followup_messages, max_tokens=512, temperature=0.1)
        if _is_fallback_text(final_text):
            return str(result), True, tool_trace
        return final_text, True, tool_trace

    return text, False, tool_trace


def finalize_answer(model_text, retrieved_items, tool_used=False, tool_trace=None):
    text = model_text.strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?])", r"\1", text)

    if not text:
        return FALLBACK

    if FALLBACK.lower() in text.lower():
        return FALLBACK

    if tool_used:
        if tool_trace:
            try:
                trace_json = json.dumps(tool_trace, ensure_ascii=False, indent=2)
            except Exception:
                trace_json = str(tool_trace)
            return f"{text}\n\nTool Trace:\n```json\n{trace_json}\n```"
        return text

    if not retrieved_items:
        return FALLBACK

    if _CITATION_PATTERN.search(text):
        return text

    sources_list = []
    seen = set()
    for item in retrieved_items:
        c = _citation_for_item(item)
        if c not in seen:
            seen.add(c)
            sources_list.append(c)

    return f"{text}\n\nSources: {', '.join(sources_list)}"


def _local_chat(llm, messages, max_tokens=512, temperature=0.1):
    if hasattr(llm, "create_chat_completion"):
        res = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return res["choices"][0]["message"]["content"]

    prompt = _render_messages(messages)
    res = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["User:", "Assistant:", "<|im_end|>"],
    )
    return res["choices"][0]["text"]


def _render_messages(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    if "TOOL_CALL" not in text:
        return None

    after = text.split("TOOL_CALL", 1)[1].strip()
    brace_index = after.find("{")
    if brace_index == -1:
        return None

    raw = after[brace_index:]
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict) or "name" not in obj:
        return None

    obj["raw"] = f"TOOL_CALL {json.dumps(obj, ensure_ascii=False)}"
    return obj


def _normalize_tool_args(name: str, args: Any) -> Dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        if name == "web_search":
            return {"query": args}
        if name == "get_weather":
            return {"location": args}
        if name == "get_country_info":
            return {"country": args}
    return {}
