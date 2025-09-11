# serve.py
from pydantic import BaseModel
from typing import List, Optional, Literal
from rag_graph import build_graph
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uuid
from langchain_core.messages import AIMessageChunk
import json


app = FastAPI()
rag_graph = build_graph()

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = True



@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    question = next((m.content for m in request.messages[::-1] if m.role == "user"), None)

    if not question:
        return {"error": "No user message found."}
    history = [m.content for m in request.messages if m.role != "user"]

    initial_state = {"question": question, "documents": [], "urls": [], "answer": "", "delta": "", "stream": request.stream, "history": history, "counter": 0}

    if request.stream:
        async def event_stream():
            async for item in rag_graph.astream(initial_state, stream_mode="messages"):  # Specify which keys to stream):
                message, metadata = item
                if isinstance(message, AIMessageChunk):
                    content = message.content
                    chunk = {
                        "choices": [{
                            "id": str(uuid.uuid4()),
                            "delta": {"content": content},
                            "finish_reason": None,
                            "index": 0
                        }],
                        "object": "chat.completion.chunk"
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    else:
        final_state = await rag_graph.ainvoke(initial_state)
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": final_state["answer"]
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "model": request.model,
        }


# Add your custom model list here
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "ROCm-RAG-Langgraph",
                "object": "model",
                "owned_by": "AMD",
            }
        ]
    }


