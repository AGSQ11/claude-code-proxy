"""Google Generative AI client for native genai API format."""

import asyncio
import json
import logging
import os
import traceback
from typing import Optional, AsyncGenerator, Dict, Any

from fastapi import HTTPException

try:
    from google import genai
except ImportError:
    genai = None  # Will be checked when creating client


logger = logging.getLogger(__name__)


class GoogleGenAIClient:
    """Google Generative AI client with native genai API support."""

    def __init__(self, api_key: str, timeout: int = 90):
        """Initialize Google GenAI client.

        Args:
            api_key: Google API key from AI Studio
            timeout: Request timeout in seconds
        """
        if genai is None:
            raise ImportError(
                "google-genai package is not installed. "
                "Install with: pip install google-genai"
            )

        self.api_key = api_key
        self.timeout = timeout

        # Set API key as environment variable - the SDK reads from GEMINI_API_KEY or GOOGLE_API_KEY
        # We set both the env var AND pass as parameter to ensure it's used
        os.environ['GOOGLE_API_KEY'] = api_key
        os.environ['GEMINI_API_KEY'] = api_key

        # Create the genai client
        # IMPORTANT: Pass both api_key parameter AND vertexai=False
        # - vertexai=False ensures we use Developer API endpoint (generativelanguage.googleapis.com)
        # - api_key parameter ensures API key authentication is used (not OAuth2)
        self.client = genai.Client(api_key=api_key, vertexai=False)

        self.active_requests: Dict[str, asyncio.Event] = {}

    async def create_chat_completion(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a non-streaming chat completion using Google's native API.

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

            # Call Google API (blocking, so run in thread pool)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=contents,
            )

            # Convert response to OpenAI format
            return self._convert_response_to_openai(response, request)

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
        """Create a streaming chat completion using Google's native API.

        Note: Since this is used with StreamingResponse, we must ALWAYS yield error events
        in SSE format instead of raising HTTPException.

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

            # Stream from Google API (blocking, so run in thread pool)
            def _generate_stream():
                return self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                )

            stream = await asyncio.to_thread(_generate_stream)

            # Convert stream to OpenAI SSE format
            async for chunk in self._stream_to_openai_format(stream, model, request_id):
                yield chunk

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
            List of content strings for Google API
        """
        # For simplicity, combine all messages into a single prompt
        # Google's genai SDK handles multi-turn differently, but this works for basic use
        content_parts = []

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

            if role == "system":
                content_parts.append(f"System: {content_text}")
            elif role == "user":
                content_parts.append(content_text)
            elif role == "assistant":
                content_parts.append(f"Assistant: {content_text}")

        combined = "\n\n".join(content_parts) if content_parts else "Hello"
        # Return as list of strings (Google API expects list)
        return [combined]

    def _convert_response_to_openai(
        self, response: Any, original_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert Google response to OpenAI format.

        Args:
            response: Google GenAI response
            original_request: Original OpenAI-format request

        Returns:
            OpenAI-format response dict
        """
        # Extract text from Google response
        text = response.text if hasattr(response, "text") else ""

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
                "prompt_tokens": getattr(
                    response, "usage_metadata", {}
                ).get("prompt_token_count", 0)
                if hasattr(response, "usage_metadata")
                else 0,
                "completion_tokens": getattr(
                    response, "usage_metadata", {}
                ).get("candidates_token_count", 0)
                if hasattr(response, "usage_metadata")
                else 0,
                "total_tokens": getattr(response, "usage_metadata", {}).get(
                    "total_token_count", 0
                )
                if hasattr(response, "usage_metadata")
                else 0,
            },
        }

    async def _stream_to_openai_format(
        self, stream: Any, model: str, request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Convert Google stream to OpenAI SSE format.

        Args:
            stream: Google GenAI stream (generator)
            model: Model name
            request_id: Optional request ID for cancellation

        Yields:
            SSE-formatted chunks
        """
        chunk_id = f"chatcmpl-google-{int(asyncio.get_event_loop().time())}"

        # Iterate through the stream (which is a blocking generator)
        # We need to run this in a thread pool to avoid blocking
        def _iterate_stream():
            chunks = []
            try:
                for chunk in stream:
                    chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error iterating stream: {e}")
            return chunks

        chunks = await asyncio.to_thread(_iterate_stream)

        for chunk in chunks:
            # Check for cancellation
            if (
                request_id
                and request_id in self.active_requests
                and self.active_requests[request_id].is_set()
            ):
                logger.info(f"Request {request_id} was cancelled, stopping stream")
                break

            # Extract text from chunk
            chunk_text = chunk.text if hasattr(chunk, "text") else ""

            if chunk_text:
                # Build OpenAI-format chunk
                chunk_data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }
                    ],
                }

                # Yield as SSE event
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

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

    async def cancel_request(self, request_id: str):
        """Cancel an ongoing request.

        Args:
            request_id: The request ID to cancel
        """
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            logger.info(f"Cancellation requested for {request_id}")
