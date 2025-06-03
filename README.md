# Research Agent (LangGraph + Ollama)

This repo contains a minimal agentic workflow that takes a research topic from the user, performs a web search using Serper API, retrieves content from different sources (news sites, Wikipedia, PDFs, generic webpages), and compiles a report using a locally hosted LLM via Ollama.

## Features

- Uses LangGraph and LangChain Core
- Uses a local LLM through Ollama
- Searches the Web through Serper
- Generates a report using local LLM (Ollama)
- Saves to pdf

## Setup
-Get your free Serper key from https://serper.dev/
-Download and install ollama and get the local language model running (ollama run "modelname") - we use gemma3:4b
-pip install requirements.txt

