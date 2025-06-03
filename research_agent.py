import requests
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict
from fpdf import FPDF
import unicodedata

SERPER_API_KEY = "enter_your_serper_api_key" # Available for free at Serper



class AgentState(TypedDict):
    topic: str
    instruction: str
    search_results: list
    summary: str
    report: str



def google_search(query: str) -> list:
    res = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY},
        json={"q": query},
    )
    res.raise_for_status()
    return res.json().get("organic", [])


def search_node(state: dict) -> dict:
    topic = state["topic"]
    results = google_search(topic)
    links = [{"title": r["title"], "link": r["link"], "snippet": r.get("snippet", "")} for r in results[:5]]
    return {**state, "search_results": links}


def summarize_node(state: dict) -> dict:
    snippets = "\n\n".join(f"{r['title']}:\n{r['snippet']}" for r in state["search_results"])
    prompt = f"Topic: {state['topic']}\n\nExtract key insights from the following:\n\n{snippets}"
    summary = call_ollama_llm(prompt)
    return {**state, "summary": summary}


def report_node(state: dict) -> dict:
    prompt = f"Write a detailed report on the topic '{state['topic']}' using this summary:\n\n{state['summary']}\n\n" \
             f"Format as per the user's instruction:\n\n{state.get('instruction', 'Plain English report')}"
    report = call_ollama_llm(prompt)
    return {**state, "report": report}


# Local Ollama wrapper (same as before)
def call_ollama_llm(prompt: str, model="gemma3:4b") -> str:
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    res.raise_for_status()
    return res.json()["response"].strip()


# Assemble LangGraph
builder = StateGraph(state_schema=AgentState)
builder.add_node("search", RunnableLambda(search_node))
builder.add_node("summarize", RunnableLambda(summarize_node))
builder.add_node("report_node", RunnableLambda(report_node))



builder.set_entry_point("search")
builder.add_edge("search", "summarize")
builder.add_edge("summarize", "report_node")
builder.add_edge("report_node", END)
graph = builder.compile()

def sanitize_text(text):
    return unicodedata.normalize("NFKD", text).encode("latin1", "ignore").decode("latin1")




if __name__ == "__main__":
    topic = input("Enter research topic: ")
    instruction = input("Enter formatting instructions (optional): ")
    result = graph.invoke({"topic": topic, "instruction": instruction})
    print("\n=== Final Report ===\n")
    print(result["report"])
    

    

    text = result["report"]
    text = sanitize_text(text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)

    file_name = topic.replace(" ", "_")+".pdf"

    pdf.output(file_name)
