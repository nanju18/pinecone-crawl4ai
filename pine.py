import asyncio
import os
import json
import openai
from dotenv import load_dotenv
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

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and configurations from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print(PINECONE_API_KEY)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
print(PINECONE_INDEX)
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
print(PINECONE_ENVIRONMENT)

if not PINECONE_API_KEY or not OPENAI_API_KEY or not PINECONE_INDEX or not PINECONE_ENVIRONMENT:
    raise ValueError("One or more environment variables are missing. Please check your .env file.")

# OpenAI API Client
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)

# Ensure the index exists
if PINECONE_INDEX not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=PINECONE_INDEX,
        dimension=1536,  # Adjust based on the embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="gcp",
            region=PINECONE_ENVIRONMENT
        )
    )

index = pinecone_client.index(PINECONE_INDEX)

# Initialize LangChain Pinecone Vector Store
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = LangchainPinecone(index, embedding_function, "text")

def clean_html(html_content: str) -> str:
    html_content = re.sub(r"(?is)<(script|style|header|footer|nav|aside|meta|[documents]).*?>.*?</\1>", "", html_content)
    html_content = re.sub(r"</?(div|p|br|h\d|li|ul|ol|blockquote|tr|td|th)>", "\n", html_content)
    html_content = re.sub(r"<[^>]+>", "", html_content)
    html_content = re.sub(r"[ \t]+", " ", html_content)
    html_content = re.sub(r"\n\s*\n+", "\n", html_content).strip()
    return html_content

def fetch_and_clean_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return clean_html(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 500) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", "", "."]
    )
    return splitter.split_text(text)

async def crawl_and_store(url: str, depth: int):
    """Crawls a website, extracts content, and stores it in Pinecone."""
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=depth, include_external=False),
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url, config=config)

        for result in results:
            cleaned_text = clean_html(result.cleaned_html) if result.cleaned_html else "No content extracted."
            metadata = {"url": result.url}

            chunks = chunk_text(cleaned_text)
            for chunk in chunks:
                vector_store.add_texts([chunk], metadatas=[metadata])

        print(f"Stored {len(results)} pages in Pinecone vector database.")

async def query_rag(question: str):
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

async def chat_system():
    """Handles user input and queries."""
    base_url = input("Enter the base URL to crawl: ")
    depth = int(input("Enter the crawl depth: "))
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
