import os
import requests
import streamlit as st
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict


# ================== CONFIG ===================
OPENWEATHER_API_KEY = "your_api_key"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ================== STATE ===================
class WeatherState(TypedDict):
    location: str
    weather_summary: str
    email_body: str



# ================== WEATHER FETCH NODE ===================
def get_weather(location: str, api_key: str) -> str:
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric"
    res = requests.get(url)
    data = res.json()
    if "list" not in data:
        raise ValueError("Weather API error: " + str(data))
    forecast = data["list"][:8]  # next ~24 hours
    return "\n".join(
        f"{f['dt_txt']}: {f['weather'][0]['description']}, {f['main']['temp']}°C"
        for f in forecast
    )

def weather_node(state: dict) -> dict:
    location = state["location"]
    summary = get_weather(location, OPENWEATHER_API_KEY)
    return {**state, "weather_summary": summary}


# ================== FORMAT NODE ===================
def call_ollama_llm(prompt: str, model=OLLAMA_MODEL) -> str:
    res = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
    )
    res.raise_for_status()
    return res.json()["response"].strip()

def format_advice(state: dict) -> dict:
    location = state["location"]
    summary = state["weather_summary"]
    prompt = f"""You are a helpful weather assistant.
Below is the weather forecast for the next several hours for {location}.
Summarize the forecast clearly and suggest how to prepare for the day.
Include advice on clothing, and whether to bring umbrella, jacket, sunscreen, etc.

Weather Forecast:
{summary}
"""
    email_text = call_ollama_llm(prompt)
    return {**state, "email_body": email_text}


# ================== BUILD GRAPH ===================
builder = StateGraph(state_schema=WeatherState)
builder.add_node("weather_node", RunnableLambda(weather_node))
builder.add_node("format_node", RunnableLambda(format_advice))
builder.set_entry_point("weather_node")
builder.add_edge("weather_node", "format_node")
builder.add_edge("format_node", END)
graph = builder.compile()


# ================== STREAMLIT APP ===================
st.title("Weather Preparation Assistant")

location = st.text_input("Enter location as city, country-code (e.g., Montreal, CA):")

if st.button("Get Forecast and Advice"):
    if not location:
        st.warning("Please enter a location.")
    else:
        try:
            with st.spinner("Generating..."):
                result = graph.invoke({"location": location})
            st.subheader("Weather Forecast")
            st.text(result["weather_summary"])
            st.subheader("Preparation Advice")
            st.write(result["email_body"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
