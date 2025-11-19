"""Google Generative AI REST API client using direct HTTP calls."""

import asyncio
import json
import logging
import traceback
from typing import Optional, AsyncGenerator, Dict, Any

import httpx
from fastapi import HTTPException


logger = logging.getLogger(__name__)


class GoogleRestClient:
    """Google Generative AI client using direct REST API calls."""

    def __init__(self, api_key: str, timeout: int = 90):
        """Initialize Google REST client.

        Args:
            api_key: Google API key from AI Studio
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        # Create HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Content-Type": "application/json",
            }
        )

        self.active_requests: Dict[str, asyncio.Event] = {}

    async def create_chat_completion(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a non-streaming chat completion using Google's REST API.

        Args:
            request: OpenAI-format request dict
            request_id: Optional request ID for cancellation tracking

        Returns:
            OpenAI-format response dict
        """
        try:
            # Track this request for potential cancellation
            if request_id:
                self.active_requests[request_id] = asyncio.Event()

            # Extract parameters
            model = request.get("model", "gemini-2.0-flash-exp")
            messages = request.get("messages", [])

            # Build contents from messages
            contents = self._convert_messages_to_contents(messages)

            # Build request body
            body = {
                "contents": contents,
                "generationConfig": {
                    "temperature": request.get("temperature", 1.0),
                    "maxOutputTokens": request.get("max_tokens", 8192),
                }
            }

            # Call Google REST API
            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

            logger.info(f"Calling Google REST API: {model}")
            response = await self.http_client.post(url, json=body)
            response.raise_for_status()

            result = response.json()

            # Convert response to OpenAI format
            return self._convert_response_to_openai(result, request)

        except httpx.HTTPStatusError as e:
            error_msg = f"Google API HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=e.response.status_code, detail=error_msg)
        except Exception as e:
            error_msg = f"Google API error: {str(e)}"
            logger.error(f"Error in Google API call: {error_msg}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    async def create_chat_completion_stream(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion using Google's REST API.

        Args:
            request: OpenAI-format request dict
            request_id: Optional request ID for cancellation tracking

        Yields:
            SSE-formatted chunks (data: {...}\n\n)
        """
        try:
            # Track this request for potential cancellation
            if request_id:
                self.active_requests[request_id] = asyncio.Event()

            # Extract parameters
            model = request.get("model", "gemini-2.0-flash-exp")
            messages = request.get("messages", [])

            # Build contents from messages
            contents = self._convert_messages_to_contents(messages)

            # Build request body
            body = {
                "contents": contents,
                "generationConfig": {
                    "temperature": request.get("temperature", 1.0),
                    "maxOutputTokens": request.get("max_tokens", 8192),
                }
            }

            # Call Google REST API with streaming
            url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"

            logger.info(f"Calling Google REST API (streaming): {model}")

            chunk_id = f"chatcmpl-google-{int(asyncio.get_event_loop().time())}"

            async with self.http_client.stream("POST", url, json=body) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # Check for cancellation
                    if (
                        request_id
                        and request_id in self.active_requests
                        and self.active_requests[request_id].is_set()
                    ):
                        logger.info(f"Request {request_id} was cancelled, stopping stream")
                        break

                    # Parse SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        try:
                            data = json.loads(data_str)

                            # Extract text from Google response
                            text = ""
                            if "candidates" in data and len(data["candidates"]) > 0:
                                candidate = data["candidates"][0]
                                if "content" in candidate and "parts" in candidate["content"]:
                                    for part in candidate["content"]["parts"]:
                                        if "text" in part:
                                            text += part["text"]

                            if text:
                                # Build OpenAI-format chunk
                                chunk_data = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE data: {data_str}")
                            continue

            # Send final chunk with finish_reason
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except httpx.HTTPStatusError as e:
            error_msg = f"Google API HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(f"Unexpected error in streaming: {error_msg}")
            logger.error(traceback.format_exc())
            # Always yield error event in SSE format (response already started)
            error_data = {"error": {"type": "api_error", "message": error_msg}}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_msg = f"Google API error: {str(e)}"
            logger.error(f"Unexpected error in streaming: {error_msg}")
            logger.error(traceback.format_exc())
            # Always yield error event in SSE format (response already started)
            error_data = {"error": {"type": "api_error", "message": error_msg}}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def _convert_messages_to_contents(self, messages: list) -> list:
        """Convert OpenAI messages to Google contents format.

        Args:
            messages: List of OpenAI-format messages

        Returns:
            List of Google API content objects
        """
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle both string and list content (for multimodal messages)
            if isinstance(content, list):
                # Extract text from content parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content_text = " ".join(text_parts)
            else:
                content_text = str(content)

            # Map roles: system -> user, assistant -> model
            google_role = "user"
            if role == "assistant":
                google_role = "model"
            elif role == "system":
                google_role = "user"
                content_text = f"System: {content_text}"

            contents.append({
                "role": google_role,
                "parts": [{"text": content_text}]
            })

        return contents

    def _convert_response_to_openai(
        self, response: Dict[str, Any], original_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert Google REST API response to OpenAI format.

        Args:
            response: Google REST API response
            original_request: Original OpenAI-format request

        Returns:
            OpenAI-format response dict
        """
        # Extract text from Google response
        text = ""
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        text += part["text"]

        # Extract usage metadata
        usage_metadata = response.get("usageMetadata", {})

        # Build OpenAI-format response
        return {
            "id": f"chatcmpl-google-{int(asyncio.get_event_loop().time())}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": original_request.get("model", "gemini-2.0-flash-exp"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0),
            },
        }

    async def cancel_request(self, request_id: str):
        """Cancel an ongoing request.

        Args:
            request_id: The request ID to cancel
        """
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            logger.info(f"Cancellation requested for {request_id}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close HTTP client."""
        await self.http_client.aclose()
