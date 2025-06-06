## Simple Agents with Local LLM (LangGraph + Ollama)

This repo contains examples of minimal agentic workflow with local LLM. 

- Uses LangGraph and LangChain Core
- Uses a local LLM through Ollama

The following agents are implemented: 

## Agents
1. research_agent.py - takes a research topic from user, performs a web search using Serper API, retrieves content from different sources and compiles a report using a locally hosted LLM via Ollama ,saves to pdf.
2. weather_email_agent.py - takes a location from User, gets the weather forecast and writes a friendly message about the weather update
3. weather_agent_app.py - same as above, but as a streamlit app
4. article_summarizer_agent.py - For a given topic summarizes 10 recent news headlines and prepares a report in pdf

## Setup

- Create an account and get your free Serper key from https://serper.dev/
- Create an account and get your free Weather API key from https://home.openweathermap.org/api_keys
- Download and install ollama and get the local language model running (ollama run "modelname"): we use gemma3:4b
- pip install requirements.txt

