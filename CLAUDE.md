# Claude Code Proxy

## Project Overview

**Claude Code Proxy** is a FastAPI-based proxy server that enables Claude Code (Anthropic's CLI tool) to work with multiple AI model providers through an OpenAI-compatible API interface. The proxy translates between Claude Code's requests (which expect OpenAI API format) and various backend providers including OpenAI, Azure OpenAI, Google Generative AI, and other OpenAI-compatible endpoints.

## Purpose

Claude Code is designed to work with models that follow the OpenAI API specification. However, many powerful models (like Google's Gemini) use different API formats. This proxy bridges that gap by:

1. **Accepting OpenAI-format requests** from Claude Code
2. **Translating requests** to the appropriate provider's format
3. **Converting responses** back to OpenAI format
4. **Managing multiple providers** across different model tiers

## Architecture

### Core Components

```
Claude Code (client)
    ↓ OpenAI API format
[FastAPI Proxy Server]
    ├── Model Mapping Layer (claude models → provider models)
    ├── Client Manager (per-tier provider clients)
    │   ├── OpenAI Client (for OpenAI/Azure/compatible APIs)
    │   └── Google GenAI Client (for Google Gemini models)
    └── Request/Response Translation
    ↓
Backend Providers (OpenAI, Google, Azure, etc.)
```

### File Structure

```
claude-code-proxy/
├── src/
│   ├── core/
│   │   ├── client.py              # OpenAI client (for OpenAI-compatible APIs)
│   │   ├── google_client.py       # Google GenAI SDK client
│   │   ├── google_rest_client.py  # Google REST API client (alternative)
│   │   ├── client_manager.py      # Multi-provider client factory
│   │   ├── config.py              # Configuration management
│   │   └── model_mapper.py        # Claude model → provider model mapping
│   ├── routes/
│   │   └── chat.py                # /v1/chat/completions endpoints
│   ├── middleware/
│   │   └── auth.py                # API key authentication
│   └── main.py                    # FastAPI application entry point
├── pyproject.toml                 # Dependencies and project metadata
└── .env                           # Configuration (not in repo)
```

## Key Features

### 1. Multi-Provider Support

Each model tier (BIG/MIDDLE/SMALL) can use a different provider:

```env
# Example: Mix Google, OpenAI, and Azure
BIG_MODEL_PROVIDER=google
BIG_MODEL=gemini-3-pro-preview
BIG_MODEL_API_KEY=your-google-api-key

MIDDLE_MODEL_PROVIDER=openai
MIDDLE_MODEL=gpt-4o
MIDDLE_MODEL_API_KEY=your-openai-key
MIDDLE_MODEL_BASE_URL=https://api.openai.com/v1

SMALL_MODEL_PROVIDER=openai
SMALL_MODEL=gpt-4o-mini
SMALL_MODEL_API_KEY=your-azure-key
SMALL_MODEL_BASE_URL=https://your-azure.openai.azure.com
SMALL_MODEL_AZURE_API_VERSION=2024-02-15-preview
```

### 2. Google Generative AI Integration

**Two Implementation Approaches:**

#### A. Google GenAI SDK (Current/Recommended)
- Uses official `google-genai` Python SDK
- Supports Vertex AI authentication with API keys
- Native Python types (`types.Content`, `types.Part`, etc.)
- Real-time streaming with background thread + queue

**Key Implementation Details:**
```python
# Client initialization
os.environ['GOOGLE_CLOUD_API_KEY'] = api_key
client = genai.Client(vertexai=True, api_key=api_key)

# Message conversion
contents = [
    types.Content(
        role="user",  # or "model" for assistant
        parts=[types.Part.from_text(text="message")]
    )
]

# Configuration
config = types.GenerateContentConfig(
    temperature=1.0,
    max_output_tokens=8192,
    system_instruction=[types.Part.from_text(text="system prompt")],
    safety_settings=[...]
)

# Streaming
stream = client.models.generate_content_stream(
    model="gemini-3-pro-preview",
    contents=contents,
    config=config
)
```

#### B. Google REST API (Alternative)
- Direct HTTP calls to Google's REST endpoints
- Two different API bases:
  - Developer API: `generativelanguage.googleapis.com/v1beta` (most models)
  - AI Platform API: `aiplatform.googleapis.com/v1/publishers/google` (Gemini 3 models)
- Useful for debugging or when SDK has issues

### 3. Model Tier System

Claude Code uses three model tiers:
- **claude-sonnet-4**: Maps to BIG_MODEL
- **claude-opus-4**: Maps to BIG_MODEL
- **claude-haiku-4**: Maps to SMALL_MODEL
- **Other models**: Default to BIG_MODEL with warning

Each tier can be independently configured with different providers.

### 4. Streaming Support

Both streaming and non-streaming requests are fully supported:

**Streaming Implementation (Google):**
```python
async def create_chat_completion_stream():
    # Create stream in thread pool (SDK is blocking)
    stream = await asyncio.to_thread(
        client.models.generate_content_stream,
        model=model,
        contents=contents,
        config=config
    )

    # Use queue + background thread for real-time streaming
    # (avoids buffering all chunks before sending)
    chunk_queue = queue.Queue()

    def _stream_reader():
        for chunk in stream:
            chunk_queue.put(("chunk", chunk))
        chunk_queue.put(("done", None))

    threading.Thread(target=_stream_reader, daemon=True).start()

    # Yield chunks as they arrive
    while True:
        item_type, item_data = await asyncio.to_thread(
            chunk_queue.get, timeout=60.0
        )
        if item_type == "done":
            break
        elif item_type == "chunk":
            chunk_text = getattr(item_data, "text", None) or ""
            if chunk_text:
                yield sse_formatted_chunk(chunk_text)
```

### 5. Request Cancellation

The proxy supports cancelling ongoing requests:
- Tracks active requests by ID
- Uses asyncio Events for cancellation signaling
- Stops streaming when cancellation is detected

## Configuration

### Environment Variables

**Proxy Server:**
```env
HOST=0.0.0.0
PORT=8000
AUTH_KEY=your-secret-key-for-claude-code
REQUEST_TIMEOUT=90
LOG_LEVEL=INFO
```

**Model Tiers (repeat for MIDDLE and SMALL):**
```env
BIG_MODEL_PROVIDER=google  # or "openai"
BIG_MODEL=gemini-3-pro-preview
BIG_MODEL_API_KEY=your-api-key
BIG_MODEL_BASE_URL=https://api.openai.com/v1  # OpenAI provider only
BIG_MODEL_AZURE_API_VERSION=2024-02-15-preview  # Azure only
```

**Legacy (backward compatibility):**
```env
OPENAI_API_KEY=fallback-key  # Used if tier-specific keys not set
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Claude Code Configuration

In your Claude Code config (typically `~/.config/claude/config.json`):

```json
{
  "api": {
    "baseURL": "http://localhost:8000/v1",
    "apiKey": "your-secret-key-for-claude-code"
  },
  "models": {
    "big": "claude-sonnet-4",
    "middle": "claude-opus-4",
    "small": "claude-haiku-4"
  }
}
```

## Technical Implementation Details

### Message Format Conversion

**OpenAI → Google:**
```python
# OpenAI format
{
    "role": "system",
    "content": "You are a helpful assistant"
}

# Converted to Google format
system_instruction = [
    types.Part.from_text(text="You are a helpful assistant")
]

# User messages
{
    "role": "user",
    "content": "Hello"
}
→
types.Content(role="user", parts=[types.Part.from_text(text="Hello")])

# Assistant messages
{
    "role": "assistant",
    "content": "Hi there"
}
→
types.Content(role="model", parts=[types.Part.from_text(text="Hi there")])
```

### Response Format Conversion

**Google → OpenAI:**
```python
# Google response
response.text = "Generated text"
response.usage_metadata = {
    "prompt_token_count": 10,
    "candidates_token_count": 20,
    "total_token_count": 30
}

# Converted to OpenAI format
{
    "id": "chatcmpl-google-1234567890",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gemini-3-pro-preview",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Generated text"},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

### Error Handling

**Streaming Errors:**
```python
# CRITICAL: In streaming mode, NEVER raise HTTPException after response starts
# Always yield SSE error events instead

try:
    async for chunk in stream:
        yield chunk
except Exception as e:
    # ✅ Correct: Yield error as SSE event
    error_data = {"error": {"type": "api_error", "message": str(e)}}
    yield f"data: {json.dumps(error_data)}\n\n"

    # ❌ Wrong: This crashes with "response already started"
    # raise HTTPException(status_code=500, detail=str(e))
```

## Known Issues and Solutions

### Issue 1: Streaming Appears "Dead" on Complex Requests

**Symptom:** Simple requests work, but complex requests (e.g., "analyze this folder") hang.

**Root Cause:** Google API returns chunks with `chunk.text = None` (metadata chunks).

**Solution:**
```python
# ❌ Wrong: Crashes on None
chunk_text = chunk.text if hasattr(chunk, "text") else ""
len(chunk_text)  # TypeError if chunk.text is None

# ✅ Correct: Handles None gracefully
chunk_text = getattr(chunk, "text", None) or ""
if chunk_text:  # Skip empty chunks
    yield chunk_text
```

### Issue 2: OAuth2 Authentication Error

**Symptom:** `401 API keys are not supported by this API`

**Root Cause:** SDK defaults to OAuth2 for Vertex AI instead of API key auth.

**Solution:**
```python
# Set environment variable BEFORE creating client
os.environ['GOOGLE_CLOUD_API_KEY'] = api_key

# Use vertexai=True with explicit api_key
client = genai.Client(vertexai=True, api_key=api_key)
```

### Issue 3: Import Error for google-genai

**Symptom:** `ImportError: cannot import name 'genai' from 'google'`

**Root Cause:** Wrong import statement or package not installed.

**Solution:**
```python
# ❌ Wrong (old package)
import google.generativeai as genai

# ✅ Correct (new package)
from google import genai
from google.genai import types
```

Install: `pip install google-genai>=1.0.0`

### Issue 4: Buffering in Streaming

**Symptom:** All chunks arrive at once instead of streaming in real-time.

**Root Cause:** Collecting all chunks before yielding.

**Solution:** Use queue + background thread pattern (see streaming implementation above).

## Development History

### Phase 1: Basic Proxy (Initial)
- FastAPI server with OpenAI passthrough
- Single provider support
- Model mapping from Claude to OpenAI models

### Phase 2: Multi-Provider Architecture
- Per-tier provider configuration
- Support for OpenAI, Azure, and compatible APIs
- Custom headers per tier
- Optional OPENAI_API_KEY with smart validation

### Phase 3: Google GenAI Integration
- Attempted REST API approach (complex endpoint logic)
- Migrated to official `google-genai` SDK
- Solved Vertex AI authentication issues
- Implemented proper message/response conversion

### Phase 4: Streaming Fixes
- Fixed "response already started" error (yield SSE instead of raise)
- Fixed buffering issue (queue + background thread)
- Fixed None chunk.text handling
- Added comprehensive debug logging

### Current Status
- ✅ Multi-provider support (OpenAI, Azure, Google)
- ✅ Full streaming support with real-time chunks
- ✅ Per-tier configuration
- ✅ Request cancellation
- ✅ Robust error handling
- ✅ Google Gemini 3 Pro support via Vertex AI

## API Endpoints

### POST /v1/chat/completions

**Request (OpenAI format):**
```json
{
  "model": "claude-sonnet-4",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 1.0,
  "max_tokens": 8192,
  "stream": false
}
```

**Response (OpenAI format):**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gemini-3-pro-preview",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hi there!"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 3,
    "total_tokens": 18
  }
}
```

**Streaming Response (SSE format):**
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1234567890,"model":"gemini-3-pro-preview","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1234567890,"model":"gemini-3-pro-preview","choices":[{"index":0,"delta":{"content":" there!"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1234567890,"model":"gemini-3-pro-preview","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Dependencies

```toml
[project]
dependencies = [
    "fastapi[standard]>=0.115.11",  # Web framework
    "uvicorn>=0.34.0",               # ASGI server
    "pydantic>=2.0.0",               # Data validation
    "python-dotenv>=1.0.0",          # Environment variables
    "openai>=1.54.0",                # OpenAI client library
    "httpx>=0.25.0",                 # Async HTTP client
    "google-genai>=1.0.0",           # Google GenAI SDK
]
```

## Running the Proxy

```bash
# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run server
python -m src.main

# Or using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

```bash
# Simple test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-auth-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": false
  }'

# Streaming test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-auth-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

## Future Enhancements

Potential improvements:
- Support for more providers (Anthropic native, Cohere, etc.)
- Caching layer for repeated requests
- Rate limiting and usage tracking
- Model fallback/retry logic
- Embedding API support
- Function calling translation
- Image/multimodal support for Google models
- Health check endpoints
- Metrics and monitoring

## Contributing

When working on this codebase:

1. **Read files before editing** - Always use Read tool before Edit
2. **Handle None values** - Google API can return `None` for chunk.text
3. **Streaming errors** - Always yield SSE events, never raise HTTPException
4. **Test both modes** - Verify streaming and non-streaming work
5. **Environment variables** - Use tier-specific keys when available
6. **Debug logging** - Use the comprehensive logging for troubleshooting

## License

MIT License - See LICENSE file for details

## Support

For issues related to:
- **Claude Code**: https://github.com/anthropics/claude-code/issues
- **This Proxy**: Create an issue in this repository
- **Google GenAI SDK**: https://github.com/googleapis/python-genai

---

**Last Updated:** 2025-11-19
**Version:** 1.0.0
**Status:** Production Ready ✅
