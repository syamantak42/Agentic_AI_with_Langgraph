import requests
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict
from fpdf import FPDF
import unicodedata
import numpy as np
import json
from datetime import date, timedelta
from sentence_transformers import SentenceTransformer, util
import trafilatura

n_days = 7
start_date = (date.today() - timedelta(days=n_days)).isoformat()
NEWS_API_KEY = "your_news_api_key"

TOP_N_RELEVANT = 100
NUM_FINAL_ARTICLES = 10

class AgentState(TypedDict):
    topic: str
    instruction: str
    articles: list  # List of {"title", "url", "content"}
    summaries: list  # List of per-article summaries
    report: str

print("[INFO] Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Model loaded.")

def news_search(query: str) -> list:
    print("[INFO] Performing NewsAPI search...")
    url = (
        f'https://newsapi.org/v2/everything?'
        f'q={query}&'
        f'from={start_date}&'
        f'sortBy=popularity&'
        f'pageSize=100&'
        f'apiKey={NEWS_API_KEY}'
    )
    response = requests.get(url)
    response.raise_for_status()
    results = response.json()
    print(f"[INFO] Retrieved {len(results['articles'])} articles.")
    return results['articles']

def extract_content(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text.split()) > 100:
                return text.strip()
    except Exception as e:
        print(f"[WARN] Content extraction failed for {url}: {e}")
    return ""

def fetch_full_articles(state: dict) -> dict:
    topic = state["topic"]
    print("[INFO] Fetching articles...")
    raw_articles = news_search(topic)

    docs = [a for a in raw_articles if a.get("description")]
    descriptions = [a["description"] for a in docs]
    print(f"[INFO] Filtering top {TOP_N_RELEVANT} relevant articles out of {len(docs)} with descriptions...")

    topic_embedding = model.encode(topic, convert_to_tensor=True)
    desc_embeddings = model.encode(descriptions, convert_to_tensor=True)

    scores = util.cos_sim(topic_embedding, desc_embeddings)[0]
    top_indices = np.argsort(scores.numpy())[::-1][:TOP_N_RELEVANT]
    selected_articles = [docs[i] for i in top_indices]

    print("[INFO] Extracting full content from selected articles...")
    enriched_articles = []
    for a in selected_articles:
        content = extract_content(a["url"])
        if content:
            a["content"] = content
            enriched_articles.append(a)
        if len(enriched_articles) == NUM_FINAL_ARTICLES:
            break

    print(f"[INFO] Collected full content for {len(enriched_articles)} articles.")
    return {**state, "articles": enriched_articles}

def summarize_each_article(state: dict) -> dict:
    print("[INFO] Summarizing each article...")
    summaries = []
    for i, article in enumerate(state["articles"], 1):
        content = article['content']
        if len(content.split()) > 1000:
            content = ' '.join(content.split()[:1000])
        prompt = f'''Summarize the article below in a dense, highly informative paragraph of 100-200 words.\nDo not omit any important information.\nArticle {i}: {article['title']}\n\n{content}'''
        print(f"[INFO] Calling LLM for article {i}...")
        summary = call_ollama_llm(prompt)
        summaries.append({
            "title": article["title"],
            "source_url": article["url"],
            "source_name": article["source"]["name"],
            "summary": summary
        })
    print("[INFO] Article summarization complete.")
    return {**state, "summaries": summaries}

def create_report(state: dict) -> dict:
    print("[INFO] Generating final report...")
    instruction = state.get("instruction", "Plain English report")
    content = "\n\n".join([
        f"{i+1}. {item['title']}\nSource: {item['source_name']}\nURL: {item['source_url']}\nSummary:\n{item['summary']}"
        for i, item in enumerate(state["summaries"])
    ])
    prompt = f'''You are an expert news editor, specializing in reporting daily news on the topic of {state['topic']}.\nBelow are the recent news articles on {state['topic']}, complete with their URL, title and summary.\nWrite a structured, coherent report summarizing the {NUM_FINAL_ARTICLES} news articles below.\nClearly mention the source the news is coming from and cite the corresponding news URLs faithfully.\nDo not include anything else.\n{instruction}\n\nrecent news articles: {content}'''
    report = call_ollama_llm(prompt)
    print("[INFO] Report generated.")
    return {**state, "report": report}

def call_ollama_llm(prompt: str, model="gemma3:4b") -> str:
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        res.raise_for_status()
        return res.json()["response"].strip()
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return "[LLM failed]"

def sanitize_text(text):
    return unicodedata.normalize("NFKD", text).encode("latin1", "ignore").decode("latin1")

builder = StateGraph(state_schema=AgentState)
builder.add_node("fetch_articles", RunnableLambda(fetch_full_articles))
builder.add_node("summarize_each", RunnableLambda(summarize_each_article))
builder.add_node("create_report", RunnableLambda(create_report))

builder.set_entry_point("fetch_articles")
builder.add_edge("fetch_articles", "summarize_each")
builder.add_edge("summarize_each", "create_report")
builder.add_edge("create_report", END)
graph = builder.compile()

if __name__ == "__main__":
    print("[INFO] Starting news summarization pipeline...")
    topic = input("Enter news topic: ")
    instruction = input("Enter formatting instructions (optional): ")
    result = graph.invoke({"topic": topic, "instruction": instruction})

    print("\n=== Final Report ===\n")
    print(result["report"])

    text = sanitize_text(result["report"])
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(topic.replace(" ", "_") + ".pdf")
