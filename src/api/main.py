import json

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from src.agent.graph import astream_agent
from src.api.schemas import ChatRequest
from src.config import OLLAMA_MODEL

app = FastAPI(title="RAG Agent API")


@app.get("/health")
async def health():
    return {"status": "ok", "model": OLLAMA_MODEL}


@app.post("/chat")
async def chat(request: ChatRequest):
    async def event_stream():
        try:
            async for event in astream_agent(request.message):
                yield {"data": json.dumps(event, ensure_ascii=False)}
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "content": str(e)})}

    return EventSourceResponse(event_stream())
