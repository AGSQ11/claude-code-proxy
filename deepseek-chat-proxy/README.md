# DeepSeek Chat Proxy

An OpenAI-compatible API wrapper for [chat.deepseek.com](https://chat.deepseek.com), allowing you to use DeepSeek's free chat interface through any OpenAI-compatible client.

## Features

- ✅ OpenAI-compatible `/v1/chat/completions` endpoint
- ✅ Streaming and non-streaming responses
- ✅ Free access to DeepSeek chat models
- ✅ Simple authentication via browser token
- ✅ No official API key required

## Quick Start

### 1. Get Your Auth Token

1. Go to https://chat.deepseek.com and log in
2. Open DevTools (Press `F12` or right-click → Inspect)
3. Navigate to: **Application** → **Local Storage** → **chat.deepseek.com**
4. Find the `userToken` key
5. Copy its value (long string starting with your email/ID)

### 2. Install Dependencies

```bash
cd deepseek-chat-proxy
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and paste your userToken
```

Your `.env` should look like:
```env
DEEPSEEK_AUTH_TOKEN="your_actual_token_here"
API_KEY="sk-deepseek-proxy"
PORT="8090"
```

### 4. Run the Proxy

```bash
python main.py
```

The proxy will start on `http://localhost:8090`

## Usage

### With claude-code-proxy

Add to your `claude-code-proxy/.env`:

```bash
# Use DeepSeek for middle tier
MIDDLE_MODEL="deepseek-chat"
MIDDLE_MODEL_API_KEY="sk-deepseek-proxy"
MIDDLE_MODEL_BASE_URL="http://localhost:8090/v1"
```

### With OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-deepseek-proxy",
    base_url="http://localhost:8090/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### With curl

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-deepseek-proxy" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Available Models

- `deepseek-chat` - General chat model
- `deepseek-coder` - Coding-focused model

## API Endpoints

- `POST /v1/chat/completions` - Chat completions (OpenAI-compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /` - API information

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_AUTH_TOKEN` | **Yes** | - | Your userToken from chat.deepseek.com localStorage |
| `API_KEY` | No | `sk-deepseek-proxy` | API key for client authentication |
| `PORT` | No | `8090` | Server port |

## How It Works

This proxy:
1. Accepts OpenAI-formatted requests
2. Authenticates with chat.deepseek.com using your browser token
3. Creates a chat session
4. Streams responses back in OpenAI SSE format
5. Handles token refresh and session management

## Important Notes

### Token Expiration

Your `DEEPSEEK_AUTH_TOKEN` may expire after some time. If you get authentication errors:
1. Log out and log back into chat.deepseek.com
2. Get a fresh `userToken` from localStorage
3. Update your `.env` file

### Rate Limits

DeepSeek's chat interface may have rate limits. If you encounter limits:
- Wait a few minutes between requests
- Consider using multiple accounts (not recommended)
- Upgrade to DeepSeek's official API for production use

### Legal Disclaimer

⚠️ **Use at your own risk**

This is a reverse-engineered proxy for educational purposes. It may:
- Violate DeepSeek's Terms of Service
- Stop working if DeepSeek changes their API
- Result in account suspension

For production use, consider:
- DeepSeek's official API: https://platform.deepseek.com/
- OpenRouter: https://openrouter.ai/deepseek/deepseek-chat (has free tier)

## Troubleshooting

### "DeepSeek client not configured"
- Make sure `DEEPSEEK_AUTH_TOKEN` is set in `.env`
- Verify the token is correct (copy from browser localStorage)

### "Invalid auth token" (401 error)
- Your token has expired - get a fresh one from chat.deepseek.com
- Make sure you copied the entire token value

### "Rate limit exceeded" (429 error)
- You're sending too many requests
- Wait a few minutes before trying again

### No response / Timeout
- DeepSeek servers may be down or slow
- Try again later
- Check https://chat.deepseek.com is accessible

## Development

### Project Structure

```
deepseek-chat-proxy/
├── main.py           # FastAPI server
├── requirements.txt  # Python dependencies
├── .env.example     # Environment template
└── README.md        # This file
```

### Testing

```bash
# Test health endpoint
curl http://localhost:8090/health

# Test chat completion
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hi"}]}'
```

## Comparison with Official API

| Feature | This Proxy (Free) | Official API (Paid) |
|---------|------------------|-------------------|
| Cost | Free | $0.14/1M tokens |
| Rate Limits | Browser limits | API tier limits |
| Reliability | Depends on chat.deepseek.com | Production SLA |
| Features | Basic chat | Full API features |
| ToS Compliance | ⚠️ Unclear | ✅ Official |

## Alternative Solutions

1. **DeepSeek Official API** - https://platform.deepseek.com/
   - Cheap ($0.14/1M tokens)
   - Official, reliable
   - Full features

2. **OpenRouter** - https://openrouter.ai/deepseek/deepseek-chat
   - Free tier: 50 messages/day
   - $10 credit: 1,000 messages/day
   - OpenAI-compatible

3. **Together.ai** - https://together.ai
   - DeepSeek models available
   - Free tier + paid plans

## License

This project is for educational purposes only. Use at your own risk.

## Credits

Inspired by:
- [xtekky/deepseek4free](https://github.com/xtekky/deepseek4free) - Reverse-engineered DeepSeek API
- [qwen-code-oai-proxy](https://github.com/aptdnfapt/qwen-code-oai-proxy) - Similar proxy for Qwen
