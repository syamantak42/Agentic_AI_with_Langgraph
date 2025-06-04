import os
import requests
import smtplib
from email.mime.text import MIMEText
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict


# OpenWeatherMap setup
OPENWEATHER_API_KEY =  "your_api_key"

# ================== STATE DEFINITION ===================

class WeatherState(TypedDict):
    weather_summary: str
    email_body: str


# ================== NODE: Weather Fetcher ===================

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


def weather_node(_: dict) -> dict:
    summary = get_weather(location, OPENWEATHER_API_KEY)
    return {"weather_summary": summary}


# ================== NODE: Format Email Using LLM ===================

def call_ollama_llm(prompt: str, model="gemma3:4b") -> str:
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    res.raise_for_status()
    return res.json()["response"].strip()


def format_advice(state: dict) -> dict:
    summary = state["weather_summary"]
    prompt = f"""You are a helpful weather assistant. 
    Below is the weather forecast for the next several hours for {location}. 
    Based on the following weather forecast, write a short, friendly email, 
     - Summarize the weathe rforecast in a more easy to understand language. Do not omit any detail. 
     - Advice on how to prepare for the day. 
     - Include suggestions for clothing, and whether to bring an umbrella, jacket, sunscreen, etc.

Weather Forecast:
{summary}
"""
    email_text = call_ollama_llm(prompt)
    return {**state, "email_body": email_text}




# ================== BUILD LANGGRAPH ===================

builder = StateGraph(state_schema=WeatherState)

builder.add_node("weather_node", RunnableLambda(weather_node))
builder.add_node("format_node", RunnableLambda(format_advice))
#builder.add_node("email_node", RunnableLambda(send_email_node))

builder.set_entry_point("weather_node")
builder.add_edge("weather_node", "format_node")
#builder.add_edge("format_node", "email_node")
builder.add_edge("format_node", END)

graph = builder.compile()


# ================== RUN ===================

if __name__ == "__main__":
    location = input("Enter location as city, country code: ")
    print(f"Running daily weather agent for {location}...")
    result = graph.invoke({})
    print(result["email_body"])
    
