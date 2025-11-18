"""
DeepSeek Chat Proxy - OpenAI-compatible API wrapper for chat.deepseek.com

This proxy wraps the DeepSeek chat interface to provide an OpenAI-compatible API.
Requires authentication token from browser localStorage.
"""

import asyncio
import json
import os
import time
import uuid
from typing import Optional, AsyncGenerator, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="DeepSeek Chat Proxy", version="1.0.0")

# Configuration
DEEPSEEK_API_BASE = "https://chat.deepseek.com/api/v0"
AUTH_TOKEN = os.getenv("DEEPSEEK_AUTH_TOKEN", "")
API_KEY = os.getenv("API_KEY", "sk-deepseek-proxy")  # For client validation
PORT = int(os.getenv("PORT", "8090"))

if not AUTH_TOKEN:
    print("⚠️  WARNING: DEEPSEEK_AUTH_TOKEN not set!")
    print("   Get your token from: chat.deepseek.com > DevTools > Application > LocalStorage > userToken")


# OpenAI-compatible request models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 4096
    top_p: Optional[float] = 1.0


class DeepSeekClient:
    """Client for interacting with DeepSeek chat API"""

    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self.base_url = DEEPSEEK_API_BASE
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

    async def create_chat_session(self) -> str:
        """Create a new chat session and return chat_id"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/new",
                    headers=self.headers,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("chat_id", str(uuid.uuid4()))
            except Exception as e:
                print(f"Failed to create chat session: {e}")
                # Return a temporary ID if session creation fails
                return f"temp_{uuid.uuid4()}"

    async def chat_completion_stream(
        self, chat_id: str, messages: List[Message], model: str = "deepseek_chat"
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion responses"""

        # Build the prompt from messages
        prompt = self._build_prompt(messages)

        payload = {
            "message": prompt,
            "model_class": model,
            "temperature": 1.0,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=90.0,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    break

                                try:
                                    data = json.loads(data_str)
                                    # Extract text content
                                    if "choices" in data:
                                        for choice in data["choices"]:
                                            if "delta" in choice and "content" in choice["delta"]:
                                                yield choice["delta"]["content"]
                                    elif "text" in data:
                                        yield data["text"]
                                    elif "content" in data:
                                        yield data["content"]
                                except json.JSONDecodeError:
                                    continue

            except httpx.HTTPStatusError as e:
                error_msg = f"DeepSeek API error: {e.response.status_code}"
                if e.response.status_code == 401:
                    error_msg = "Invalid auth token. Get new token from chat.deepseek.com localStorage"
                elif e.response.status_code == 429:
                    error_msg = "Rate limit exceeded. Please wait and try again."
                raise HTTPException(status_code=e.response.status_code, detail=error_msg)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    def _build_prompt(self, messages: List[Message]) -> str:
        """Build a prompt from message history"""
        # Simple concatenation for now - can be improved
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        # Return just the last user message for simplicity
        # DeepSeek API handles context differently
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return messages[-1].content if messages else ""


# Global client instance
deepseek_client = DeepSeekClient(AUTH_TOKEN) if AUTH_TOKEN else None


def validate_api_key(authorization: Optional[str] = Header(None)):
    """Validate API key from Authorization header"""
    if not API_KEY or API_KEY == "sk-deepseek-proxy":
        return  # Skip validation if no key set

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """OpenAI-compatible chat completions endpoint"""

    if not deepseek_client:
        raise HTTPException(
            status_code=500,
            detail="DeepSeek client not configured. Set DEEPSEEK_AUTH_TOKEN in .env",
        )

    # Validate API key
    validate_api_key(authorization)

    # Create a chat session
    chat_id = await deepseek_client.create_chat_session()

    if request.stream:
        # Streaming response
        return StreamingResponse(
            stream_response(chat_id, request),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        return await non_stream_response(chat_id, request)


async def stream_response(
    chat_id: str, request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Generate streaming response in OpenAI SSE format"""

    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    full_content = ""

    try:
        async for content_chunk in deepseek_client.chat_completion_stream(
            chat_id, request.messages, request.model
        ):
            full_content += content_chunk

            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content_chunk, "role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }

            yield f"data: {json.dumps(chunk)}\n\n"

        # Send final chunk
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "api_error",
                "code": "internal_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def non_stream_response(chat_id: str, request: ChatCompletionRequest) -> Dict[str, Any]:
    """Generate non-streaming response in OpenAI format"""

    full_content = ""

    try:
        async for content_chunk in deepseek_client.chat_completion_stream(
            chat_id, request.messages, request.model
        ):
            full_content += content_chunk

        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": len(full_content.split()),
                "total_tokens": sum(len(m.content.split()) for m in request.messages)
                + len(full_content.split()),
            },
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
            },
            {
                "id": "deepseek-coder",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
            },
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if deepseek_client else "not_configured",
        "auth_configured": bool(AUTH_TOKEN),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DeepSeek Chat Proxy v1.0.0",
        "status": "running",
        "auth_configured": bool(AUTH_TOKEN),
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
        },
        "note": "Set DEEPSEEK_AUTH_TOKEN in .env to use this proxy",
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 DeepSeek Chat Proxy")
    print(f"   Auth Token: {'✅ Configured' if AUTH_TOKEN else '❌ Not Set'}")
    print(f"   Starting on http://0.0.0.0:{PORT}")
    print("")
    if not AUTH_TOKEN:
        print("⚠️  To get your auth token:")
        print("   1. Go to https://chat.deepseek.com")
        print("   2. Open DevTools (F12)")
        print("   3. Go to Application > Local Storage > chat.deepseek.com")
        print("   4. Copy the value of 'userToken'")
        print("   5. Add to .env: DEEPSEEK_AUTH_TOKEN=<your_token>")
        print("")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
