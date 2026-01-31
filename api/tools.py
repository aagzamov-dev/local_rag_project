import json
import requests

try:
    from duckduckgo_search import DDGS
except ImportError:
    # Fallback or alias handling if package name changed fully
    from ddgs import DDGS


def web_search(query: str):
    """
    Search the web for real-time information.
    Use this for news, current events, or facts not in your knowledge base.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Search Error: {str(e)}"


def get_weather(location: str):
    """
    Get current weather for a specific location (city name).
    """
    try:
        # 1. Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()
        if not geo_res.get("results"):
            return f"Could not find location: {location}"

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        name = geo_res["results"][0]["name"]

        # 2. Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
        w_res = requests.get(weather_url).json()

        current = w_res.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")

        return f"Weather in {name}: {temp}°C, Wind: {wind} km/h"
    except Exception as e:
        return f"Weather Error: {str(e)}"


def get_country_info(country: str):
    """
    Get basic facts about a country (population, capital, region, etc).
    """
    try:
        url = f"https://restcountries.com/v3.1/name/{country}?fullText=true"
        res = requests.get(url)
        if res.status_code != 200:
            # Try partial match if full fails
            url = f"https://restcountries.com/v3.1/name/{country}"
            res = requests.get(url)
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

To use a tool, you MUST format your output exactly like this:
req_tool: tool_name(argument)

Examples:
User: "What is the weather in Paris?"
Assistant: req_tool: get_weather("Paris")

User: "Who is the president of Brazil?"
Assistant: req_tool: web_search("President of Brazil")

User: "Tell me about Japan."
Assistant: req_tool: get_country_info("Japan")

Do NOT output anything else when requesting a tool.
Wait for the observation before answering.
"""
