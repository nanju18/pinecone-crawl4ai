import asyncio
import os
import json
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LangchainPinecone
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from bs4 import BeautifulSoup
from typing import List
import re
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from urllib.parse import urlencode

# API keys and configurations
PINECONE_API_KEY = "pcsk_6PUKb9_xKZ7sAuAahyMW5ZWWKB2f2929rVuRCfV6aMvugo38quhF8c4w4nZaZfQUzpfsH" # Replace with your actual Pinecone API key
OPENAI_API_KEY = "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"      # Replace with your actual OpenAI API key
PINECONE_INDEX = "pydantic1"          # Replace with your Pinecone index name
PINECONE_ENVIRONMENT = "us-west1"                # Replace with your Pinecone environment (e.g., "us-west1")
SCRAPER_API_KEY = "7ebad43756ae1fe389f8fbd957716985" # Replace with your actual ScraperAPI key
# OpenAI API Client
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists
if PINECONE_INDEX not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=PINECONE_INDEX,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region=PINECONE_ENVIRONMENT)
    )

index = pinecone_client.Index(PINECONE_INDEX)

# Initialize LangChain Pinecone Vector Store
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = LangchainPinecone(index, embedding_function, "text")

def clean_html(html_content: str) -> str:
    html_content = re.sub(r"(?is)<(script|style|header|footer|nav|aside).*?>.*?</\1>", "", html_content)
    html_content = re.sub(r"</?(div|p|br|h\d|li|ul|ol|blockquote|tr|td|th)>", "\n", html_content)
    html_content = re.sub(r"<[^>]+>", "", html_content)
    html_content = re.sub(r"[ \t]+", " ", html_content)
    html_content = re.sub(r"\n\s*\n+", "\n", html_content).strip()
    return html_content

# Fetch URL content using ScraperAPI
def fetch_and_clean_url(url: str) -> str:
    try:
        payload = {
            'api_key': SCRAPER_API_KEY,
            'url': url
        }
        response = requests.get('https://api.scraperapi.com/', params=payload, timeout=120)
        response.raise_for_status()
        return clean_html(response.content.decode('utf-8'))  # Decode and clean the content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 500) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", "", "."]
    )
    return splitter.split_text(text)

def clear_pinecone_index():
    """Clears the existing data in the Pinecone index."""
    try:
        index.delete(delete_all=True)
        print("Pinecone index cleared.")
    except Exception as e:
        print(f"Error clearing Pinecone index: {e}")

async def crawl_and_store(url: str, depth: int):
    """Crawls a website, extracts content, and stores it in Pinecone."""
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=depth, include_external=False),
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url, config=config)

        for result in results:
            cleaned_text = fetch_and_clean_url(result.url)  # Fetch content via ScraperAPI
            #print(f"\nCleaned HTML for {result.url}:\n")
            #print(cleaned_text)
            metadata = {"url": result.url}

            chunks = chunk_text(cleaned_text)
            
            for chunk in chunks:
                vector_store.add_texts([chunk], metadatas=[metadata])

        print(f"Stored {len(results)} pages in Pinecone vector database.")

chat_memory = [] 

async def query_rag(question: str):
    ''''
    """Retrieves relevant documents from Pinecone and generates an answer using OpenAI."""
    results = vector_store.similarity_search(question, k=3)

    if not results:
        return "No relevant documents found.", None

    combined_summary = "\n\n".join([doc.page_content for doc in results])
    unique_sources = list(set(doc.metadata.get("url", "Unknown source") for doc in results))

    prompt = f"""
    You are an AI assistant. Answer the following question based on the retrieved summaries.
    
    Context:
    {combined_summary}
    
    Question: {question}
    """

    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, unique_sources
'''
  # Global variable to store conversation history
    """Retrieves relevant documents from Pinecone and generates an answer using OpenAI."""
    results = vector_store.similarity_search(question, k=3)

    if not results:
        return "No relevant documents found.", None

    combined_summary = "\n\n".join([doc.page_content for doc in results])
    unique_sources = list(set(doc.metadata.get("url", "Unknown source") for doc in results))

    # Format conversation history (last 5 exchanges)
    chat_history = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_memory[-5:]])

    prompt = f"""
    You are an AI assistant. Answer the following question based on the retrieved summaries.
    
    Conversation history:
    {chat_history}
    
    Context:
    {combined_summary}
    
    Question: {question}
    """

    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # Store the conversation in memory
    chat_memory.append((question, answer))

    return answer, unique_sources

async def chat_system():
    """Handles user input and queries."""
    base_url = input("Enter the base URL to crawl: ")
    depth = int(input("Enter the crawl depth: "))
    
    clear_pinecone_index()
    
    print("Crawling and storing data, please wait...")
    await crawl_and_store(base_url, depth)
    print("Crawling complete! You can now ask questions.")

    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        answer, sources = await query_rag(user_query)
        print("\nAI Response:", answer)
        if sources:
            print("Sources:", ", ".join(sources))

if __name__ == "__main__":
    asyncio.run(chat_system())
