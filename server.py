
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from pin import crawl_and_store, query_rag

class CrawlRequest(BaseModel):
    url: str
    depth: int


class QueryRequest(BaseModel):
    question: str

app = FastAPI(title="Crawl4AI RAG API", version="1.0")
# FastAPI Endpoints
@app.post("/crawl/")
async def crawl_website(request: CrawlRequest):
    """Endpoint to start crawling a website."""
    try:
        result = await crawl_and_store(request.url, request.depth)
        return {"message": "Crawling started successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/")
async def query_system(request: QueryRequest):
    """Endpoint to retrieve information using RAG."""
    try:
        answer, sources = await query_rag(request.question)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
