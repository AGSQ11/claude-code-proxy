"""Google Generative AI client for native genai API format."""

import asyncio
import json
import logging
import traceback
from typing import Optional, AsyncGenerator, Dict, Any

from fastapi import HTTPException

try:
    import google.generativeai as genai
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
                "google-generativeai package is not installed. "
                "Install with: pip install google-generativeai"
            )

        self.api_key = api_key
        self.timeout = timeout

        # Configure the genai library with API key
        genai.configure(api_key=api_key)

        self.active_requests: Dict[str, asyncio.Event] = {}

    async def create_chat_completion(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send chat completion to Google GenAI API.

        Args:
            request: OpenAI-format request dict
            request_id: Optional request ID for cancellation tracking

        Returns:
            OpenAI-format response dict
        """
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Convert OpenAI format to Google format
            model = request.get("model", "gemini-2.0-flash-exp")
            messages = request.get("messages", [])

            # Build contents from messages
            contents = self._convert_messages_to_contents(messages)

            # Create model and generate content
            model_instance = genai.GenerativeModel(model)
            response = await asyncio.to_thread(
                model_instance.generate_content,
                contents,
            )

            # Convert response to OpenAI format
            return self._convert_response_to_openai(response, request)

        except Exception as e:
            logger.error(f"Google GenAI API error: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Google API error: {str(e)}"
            )

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    async def create_chat_completion_stream(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to Google GenAI API.

        Args:
            request: OpenAI-format request dict
            request_id: Optional request ID for cancellation tracking

        Yields:
            SSE-formatted response chunks
        """
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Convert OpenAI format to Google format
            model = request.get("model", "gemini-2.0-flash-exp")
            messages = request.get("messages", [])

            # Build contents from messages
            contents = self._convert_messages_to_contents(messages)

            # Create model and stream content
            model_instance = genai.GenerativeModel(model)

            # Generate content stream (this is a blocking call, so run in thread)
            def _generate_stream():
                return model_instance.generate_content(contents, stream=True)

            stream = await asyncio.to_thread(_generate_stream)

            # Convert stream to OpenAI SSE format
            async for chunk in self._stream_to_openai_format(stream, model, request_id):
                yield chunk

            # Signal end of stream
            yield "data: [DONE]"

        except Exception as e:
            error_msg = f"Google API error: {str(e)}"
            logger.error(f"Unexpected error in streaming: {error_msg}")
            logger.error(traceback.format_exc())
            # Always yield error event in SSE format (response already started)
            error_data = {"error": {"type": "api_error", "message": error_msg}}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}"

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def _convert_messages_to_contents(self, messages: list) -> str:
        """Convert OpenAI messages to Google contents format.

        Args:
            messages: List of OpenAI-format messages

        Returns:
            Combined content string for Google API
        """
        # For simplicity, combine all messages into a single prompt
        # Google's genai SDK handles multi-turn differently, but this works for basic use
        content_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                content_parts.append(f"System: {content}")
            elif role == "user":
                content_parts.append(content)
            elif role == "assistant":
                content_parts.append(f"Assistant: {content}")

        return "\n\n".join(content_parts) if content_parts else "Hello"

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
            "id": f"chatcmpl-{response.model_version if hasattr(response, 'model_version') else 'google'}",
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
                logger.info(f"Request {request_id} cancelled by client")
                error_data = {
                    "error": {
                        "type": "cancelled",
                        "message": "Request cancelled by client",
                    }
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}"
                return

            # Extract text from chunk
            text = chunk.text if hasattr(chunk, "text") else ""

            if text:
                # Build OpenAI-format chunk
                openai_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text, "role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }

                yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}"

        # Send final chunk with finish_reason
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}"

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide error guidance for Google API issues.

        Args:
            error_detail: Error details

        Returns:
            User-friendly error message
        """
        error_str = str(error_detail).lower()

        # API key issues
        if "invalid api key" in error_str or "unauthorized" in error_str:
            return "Invalid Google API key. Get a key from https://aistudio.google.com/apikey"

        # Rate limiting
        if "rate limit" in error_str or "quota" in error_str:
            return "Google API rate limit exceeded. Please wait and try again."

        # Model not found
        if "model" in error_str and (
            "not found" in error_str or "does not exist" in error_str
        ):
            return "Model not found. Check your model name (e.g., gemini-2.0-flash-exp)"

        # Default: return original message
        return str(error_detail)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if request was found and cancelled
        """
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
