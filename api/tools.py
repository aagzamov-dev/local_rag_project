import json
import re
import requests

REQUEST_TIMEOUT = 10

try:
    from duckduckgo_search import DDGS
except ImportError:
    # Fallback or alias handling if package name changed fully
    from ddgs import DDGS


_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "for",
    "to",
    "from",
    "about",
    "info",
    "information",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "please",
}


def _normalize_query(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace('"', "").replace("'", "")
    cleaned = re.sub(r"[?!.]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _expand_search_queries(query: str):
    base = _normalize_query(query)
    if not base:
        return []

    tokens = [t for t in re.split(r"\s+", base) if t]
    filtered = [t for t in tokens if t.lower() not in _STOPWORDS]
    no_stop = " ".join(filtered) if filtered else base

    variants = [base, no_stop, f"\"{base}\"" if base else ""]

    lower_base = base.lower()
    if "created year" in lower_base:
        variants.extend(
            [
                lower_base.replace("created year", "founded"),
                lower_base.replace("created year", "founded year"),
                lower_base.replace("created year", "founded in"),
                lower_base.replace("created year", "established"),
                f"{no_stop} founded",
            ]
        )

    if "who is" in lower_base:
        variants.append(lower_base.replace("who is", "about"))

    # Add a focused entity variant if query is short
    if len(tokens) <= 3:
        variants.append(f"{base} official site")

    # De-duplicate while preserving order
    seen = set()
    expanded = []
    for v in variants:
        v = _normalize_query(v)
        if v and v.lower() not in seen:
            seen.add(v.lower())
            expanded.append(v)
    return expanded


def web_search(query: str):
    """
    Search the web for real-time information.
    Use this for news, current events, or facts not in your knowledge base.
    """
    try:
        with DDGS() as ddgs:
            expanded = _expand_search_queries(query)
            all_results = []
            seen_urls = set()

            for q in expanded[:5]:
                batch = list(ddgs.text(q, max_results=6))
                if not batch:
                    batch = list(ddgs.text(q, max_results=6, backend="lite"))
                if not batch:
                    batch = list(ddgs.text(q, max_results=6, backend="html"))
                for item in batch:
                    url = (item.get("href") or item.get("url") or "").strip()
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    item["query"] = q
                    all_results.append(item)
                if len(all_results) >= 12:
                    break

            payload = {
                "source": "duckduckgo",
                "query": query,
                "expanded_queries": expanded,
                "results": all_results,
            }
            return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return f"Search Error: {str(e)}"


def get_weather(location: str):
    """
    Get current weather for a specific location (city name).
    """
    try:
        # 1. Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=REQUEST_TIMEOUT).json()
        if not geo_res.get("results"):
            return f"Could not find location: {location}"

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        name = geo_res["results"][0]["name"]

        # 2. Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
        w_res = requests.get(weather_url, timeout=REQUEST_TIMEOUT).json()

        current = w_res.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")

        payload = {
            "source": "open-meteo",
            "location": {"name": name, "latitude": lat, "longitude": lon},
            "current": {
                "temperature_c": temp,
                "wind_kmh": wind,
                "humidity_percent": humidity,
            },
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return f"Weather Error: {str(e)}"


def get_country_info(country: str):
    """
    Get basic facts about a country (population, capital, region, etc).
    """
    try:
        url = f"https://restcountries.com/v3.1/name/{country}?fullText=true"
        res = requests.get(url, timeout=REQUEST_TIMEOUT)
        if res.status_code != 200:
            # Try partial match if full fails
            url = f"https://restcountries.com/v3.1/name/{country}"
            res = requests.get(url, timeout=REQUEST_TIMEOUT)
            if res.status_code != 200:
                return f"Could not find country: {country}"

        data = res.json()[0]

        info = {
            "name": data.get("name", {}).get("common"),
            "capital": data.get("capital", ["N/A"])[0],
            "population": data.get("population"),
            "region": data.get("region"),
            "subregion": data.get("subregion"),
            "languages": list(data.get("languages", {}).values()),
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Country API Error: {str(e)}"


# Registry of executable functions
AVAILABLE_TOOLS = {
    "web_search": web_search,
    "get_weather": get_weather,
    "get_country_info": get_country_info,
}


def get_openai_tools():
    """
    JSON Schema for OpenAI Tool Calling
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for up-to-date information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g. 'current apple stock price')",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name (e.g. 'London', 'New York')",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_country_info",
                "description": "Get facts about a country.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "Country name (e.g. 'France')",
                        }
                    },
                    "required": ["country"],
                },
            },
        },
    ]


def get_local_tools_prompt():
    """
    Text instructions for Local LLM (ReAct / Tool usage)
    """
    return """
You have access to the following tools:

1. web_search(query: str): Search for real-time info.
2. get_weather(location: str): Get weather for a city.
3. get_country_info(country: str): Get facts about a country.

To use a tool, you MUST output exactly one line:
TOOL_CALL {"name":"tool_name","arguments":{...}}

Examples:
User: "What is the weather in Paris?"
Assistant: TOOL_CALL {"name":"get_weather","arguments":{"location":"Paris"}}

User: "Who is the president of Brazil?"
Assistant: TOOL_CALL {"name":"web_search","arguments":{"query":"President of Brazil"}}

User: "Tell me about Japan."
Assistant: TOOL_CALL {"name":"get_country_info","arguments":{"country":"Japan"}}

Do NOT output anything else when requesting a tool.
Wait for the tool result before answering.
"""
