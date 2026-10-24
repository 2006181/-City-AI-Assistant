
import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(
    page_title="City AI Assistant",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 City AI Assistant")
st.caption("Your AI assistant for city weather, latest news, and everyday questions.")

# -------------------------
# Environment configuration
# -------------------------

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


# -------------------------
# Weather Tool
# -------------------------

@tool
def get_weather(city: str) -> str:
    """Get the current weather, temperature, and conditions for a city."""

    if not OPENWEATHER_API_KEY:
        return "Weather is unavailable because OPENWEATHER_API_KEY is not configured."

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        city_name = data["name"]

        return (
            f"Weather in {city_name}\n"
            f"Condition: {description}\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%"
        )

    except requests.HTTPError:
        return "Unable to fetch weather. Check the city name and API key."
    except requests.RequestException:
        return "Weather service is temporarily unavailable."
    except (KeyError, IndexError, ValueError):
        return "Could not read the weather data."


# -------------------------
# News Tool
# -------------------------

@tool
def get_news(city: str) -> str:
    """Find the latest news about a city and return headlines, summaries, and links."""

    if not TAVILY_API_KEY:
        return "News is unavailable because TAVILY_API_KEY is not configured."

    try:
        tavily = TavilySearch(
            api_key=TAVILY_API_KEY,
            max_results=3,
            topic="news",
            time_range="day"
        )

        response = tavily.invoke({
            "query": f"Latest news about {city}"
        })

        if isinstance(response, str):
            import json
            try:
                response = json.loads(response)
            except ValueError:
                return response

        results = response.get("results", []) if isinstance(response, dict) else []

        if not results:
            return f"No recent news found for {city}."

        news_items = []

        for index, item in enumerate(results, start=1):
            title = item.get("title", "Untitled")
            summary = item.get("content", "No summary available.")
            url = item.get("url", "")

            news_items.append(
                f"{index}. {title}\n"
                f"{summary[:250]}\n"
                f"Article: {url}"
            )

        return "\n\n".join(news_items)

    except Exception:
        return "Unable to fetch news right now. Please try again."


# -------------------------
# Clean AI response
# -------------------------

def extract_text(content):
    """Extract readable text without printing metadata or signatures."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())

            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())

        return "\n".join(parts)

    return ""


def get_final_answer(result):
    messages = result.get("messages", [])

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            if not getattr(message, "tool_calls", []):
                answer = extract_text(message.content)
                if answer:
                    return answer

    return "I couldn't generate a response. Please try again."


# -------------------------
# Sidebar settings
# -------------------------

with st.sidebar:
    st.header("Settings")

    tool_mode = st.radio(
        "Choose assistant mode",
        [
            "Chat only",
            "Weather assistant",
            "News assistant",
            "All tools"
        ],
        index=3
    )

    st.caption("Weather and news tools run only when enabled in the selected mode.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by Gemini, LangChain, OpenWeatherMap and Tavily.")


# -------------------------
# API key validation
# -------------------------

if not GEMINI_API_KEY:
    st.error(
        "Google Gemini API key is missing. Add GOOGLE_API_KEY to your .env file."
    )
    st.stop()


# -------------------------
# Chat history
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------
# Chat input and agent
# -------------------------

user_input = st.chat_input("Ask about a city, weather, news, or anything else...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    tools = []

    if tool_mode in ("Weather assistant", "All tools"):
        tools.append(get_weather)

    if tool_mode in ("News assistant", "All tools"):
        tools.append(get_news)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    timeout=60,
                    max_retries=2
                )

                agent = create_agent(
                    model=llm,
                    tools=tools,
                    system_prompt=(
                        "You are a helpful City AI Assistant. "
                        "Give clear, accurate, concise answers. "
                        "Use available tools when relevant. "
                        "Never invent weather readings or news articles. "
                        "If a required tool is unavailable, tell the user. "
                        "Do not include signatures, hidden metadata, "
                        "or unnecessary sign-offs. "
                        "For news, include the article links returned by the tool."
                    )
                )

                result = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                })

                answer = get_final_answer(result)

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as error:
        error_message = (
            "Something went wrong. Check your API key, model name, "
            "internet connection, and installed package versions."
        )

        with st.chat_message("assistant"):
            st.error(error_message)
            with st.expander("Technical details"):
                st.code(str(error))

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message
        })
