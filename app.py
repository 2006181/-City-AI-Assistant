import json
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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
# Response and interrupt helpers
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


def result_value(result):
    """Support both the normal dict result and LangChain GraphOutput."""
    return getattr(result, "value", result)


def get_final_answer(result):
    state = result_value(result)
    messages = state.get("messages", []) if isinstance(state, dict) else []

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            if not getattr(message, "tool_calls", []):
                answer = extract_text(message.content)
                if answer:
                    return answer

    return "I couldn't generate a response. Please try again."


def get_interrupts(result):
    """Return pending LangGraph interrupts across supported result shapes."""
    interrupts = getattr(result, "interrupts", None)
    if interrupts is not None:
        return list(interrupts)

    state = result_value(result)
    if isinstance(state, dict):
        return list(state.get("__interrupt__", []))

    return []


def build_agent(tool_mode):
    """Build/cache one create_agent instance for the selected sidebar mode."""
    enabled_tools = []

    if tool_mode in ("Weather assistant", "All tools"):
        enabled_tools.append(get_weather)

    if tool_mode in ("News assistant", "All tools"):
        enabled_tools.append(get_news)

    middleware = []
    if enabled_tools:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    selected_tool.name: {
                        "allowed_decisions": ["approve", "reject"]
                    }
                    for selected_tool in enabled_tools
                },
                description_prefix="The assistant wants to run this tool",
            )
        )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        timeout=60,
        max_retries=2
    )

    return create_agent(
        model=llm,
        tools=enabled_tools,
        middleware=middleware,
        checkpointer=st.session_state.checkpointer,
        system_prompt=(
            "You are a helpful City AI Assistant. "
            "Give clear, accurate, concise answers. "
            "Use available tools when relevant. "
            "Never invent weather readings or news articles. "
            "If a required tool is unavailable, tell the user. "
            "Do not include signatures, hidden metadata, or unnecessary sign-offs. "
            "For news, include the article links returned by the tool."
        )
    )


def save_agent_result(result, thread_id, tool_mode):
    """Save a pause for review, or append the completed assistant response."""
    interrupts = get_interrupts(result)

    if interrupts:
        st.session_state.pending_approval = {
            "thread_id": thread_id,
            "tool_mode": tool_mode,
            "interrupts": interrupts,
        }
        return

    answer = get_final_answer(result)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.pending_approval = None


# -------------------------
# Session state
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()

if "agents" not in st.session_state:
    st.session_state.agents = {}

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None


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
        index=3,
        disabled=st.session_state.pending_approval is not None,
    )

    st.caption("Weather and news tools run only when enabled in the selected mode.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_approval = None
        st.session_state.agents = {}
        st.session_state.checkpointer = InMemorySaver()
        st.rerun()

    st.divider()
    st.caption("Powered by Gemini, LangChain, OpenWeatherMap and Tavily.")


# -------------------------
# API key validation
# -------------------------

if not GEMINI_API_KEY:
    st.error("Google Gemini API key is missing. Add GOOGLE_API_KEY to your .env file.")
    st.stop()


# -------------------------
# Chat history
# -------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------
# Human approval panel
# -------------------------

pending = st.session_state.pending_approval

if pending:
    with st.chat_message("assistant"):
        st.warning("The assistant is waiting for your decision before running a tool.")

        action_requests = []
        for interrupt in pending["interrupts"]:
            value = getattr(interrupt, "value", {})
            if not isinstance(value, dict):
                continue

            if isinstance(value.get("action_requests"), list):
                action_requests.extend(value["action_requests"])
            elif value.get("name"):
                # Also handle the per-call interrupt shape from newer LangChain versions.
                action_requests.append(value)

        for index, action in enumerate(action_requests, start=1):
            st.markdown(f"**Tool request {index}: `{action.get('name', 'unknown')}`**")
            st.caption(action.get("description", "Review the arguments before deciding."))
            st.json(action.get("args", {}))

        if not action_requests:
            st.error("Could not read the pending tool request. Clear the chat and try again.")
        else:
            approve_col, reject_col = st.columns(2)
            approve_clicked = approve_col.button(
                "Approve all calls",
                type="primary",
                use_container_width=True,
                key="approve_pending_tools",
            )
            reject_clicked = reject_col.button(
                "Reject all calls",
                use_container_width=True,
                key="reject_pending_tools",
            )

            if approve_clicked or reject_clicked:
                decision_type = "approve" if approve_clicked else "reject"
                decisions = [
                    {"type": decision_type}
                    if decision_type == "approve"
                    else {
                        "type": "reject",
                        "message": "The user rejected this tool call.",
                    }
                    for _ in action_requests
                ]

                agent = st.session_state.agents[pending["tool_mode"]]
                config = {
                    "configurable": {"thread_id": pending["thread_id"]}
                }

                try:
                    with st.spinner("Continuing the assistant..."):
                        # The agent resumes from its saved interrupt. Rejected calls
                        # receive a rejection ToolMessage and are never executed.
                        result = agent.invoke(
                            Command(resume={"decisions": decisions}),
                            config=config,
                        )
                        save_agent_result(
                            result,
                            pending["thread_id"],
                            pending["tool_mode"],
                        )
                    st.rerun()
                except Exception as error:
                    st.error("The assistant could not continue after that decision.")
                    with st.expander("Technical details"):
                        st.code(str(error))


# -------------------------
# Chat input and agent
# -------------------------

user_input = st.chat_input(
    "Ask about a city, weather, news, or anything else...",
    disabled=st.session_state.pending_approval is not None,
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        if tool_mode not in st.session_state.agents:
            st.session_state.agents[tool_mode] = build_agent(tool_mode)

        agent = st.session_state.agents[tool_mode]
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Each turn gets a fresh checkpoint thread and receives the visible chat
        # history. This keeps mode changes from mixing incompatible tool graphs.
        conversation = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = agent.invoke({"messages": conversation}, config=config)

            save_agent_result(result, thread_id, tool_mode)

            if st.session_state.pending_approval:
                # This panel appears earlier in the script, so rerun to render
                # the newly saved interrupt and its approval controls.
                st.rerun()
            else:
                st.markdown(st.session_state.messages[-1]["content"])

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

