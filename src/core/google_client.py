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
    from google.genai import types
except ImportError:
    genai = None  # Will be checked when creating client
    types = None


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

        # Set API key as GOOGLE_CLOUD_API_KEY environment variable (required for Vertex AI)
        os.environ['GOOGLE_CLOUD_API_KEY'] = api_key

        # Create the genai client with Vertex AI enabled
        # IMPORTANT: vertexai=True is required for models like gemini-3-pro-preview
        self.client = genai.Client(
            vertexai=True,
            api_key=api_key
        )

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

            # Build contents and config from messages
            contents, system_instruction = self._convert_messages_to_contents(messages)
            config = self._build_config(request, system_instruction)

            # Call Google API (blocking, so run in thread pool)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
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
            logger.info(f"🚀 Starting streaming request for model={model}, {len(messages)} messages")

            # Build contents and config from messages
            contents, system_instruction = self._convert_messages_to_contents(messages)
            config = self._build_config(request, system_instruction)
            logger.info(f"📋 Converted {len(contents)} contents, system_instruction={'present' if system_instruction else 'none'}")

            # Stream from Google API (blocking, so run in thread pool)
            def _generate_stream():
                logger.info(f"🔌 Calling Google API generate_content_stream for {model}...")
                result = self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                )
                logger.info(f"✅ Google API stream object created: {type(result)}")
                return result

            logger.info("⏳ Awaiting stream creation in thread pool...")
            stream = await asyncio.to_thread(_generate_stream)
            logger.info(f"✅ Stream created successfully, type={type(stream)}")

            # Convert stream to OpenAI SSE format
            logger.info("🔄 Starting stream-to-OpenAI conversion...")
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

    def _convert_messages_to_contents(self, messages: list) -> tuple[list, Optional[list]]:
        """Convert OpenAI messages to Google contents format.

        Args:
            messages: List of OpenAI-format messages

        Returns:
            Tuple of (contents list, system_instruction list)
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # System messages become system instructions
                content_text = str(content) if content else ""
                if content_text.strip():
                    system_instruction = [types.Part.from_text(text=content_text)]

            elif role == "user":
                # User messages
                content_text = str(content) if content else ""
                if content_text.strip():
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content_text)]
                    ))

            elif role == "assistant":
                # Assistant messages - may contain text, tool_calls, or both
                parts = []

                # Extract text content
                content_text = str(content) if content else ""
                if content_text.strip():
                    parts.append(types.Part.from_text(text=content_text))

                # Convert tool_calls to function_call parts
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tool_call in msg["tool_calls"]:
                        if tool_call.get("type") == "function":
                            func = tool_call.get("function", {})
                            # Parse arguments JSON string to dict
                            args_str = func.get("arguments", "{}")
                            try:
                                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            except json.JSONDecodeError:
                                logger.warning(f"⚠️ Failed to parse tool arguments: {args_str}")
                                args = {}

                            # Create FunctionCall part
                            parts.append(types.Part.from_function_call(
                                name=func.get("name", ""),
                                args=args
                            ))
                            logger.info(f"✅ Converted assistant tool_call '{func.get('name')}' to Google function_call")

                if parts:
                    contents.append(types.Content(role="model", parts=parts))

            elif role == "tool":
                # Tool result messages - convert to function_response
                tool_call_id = msg.get("tool_call_id", "")
                result_content = str(content) if content else ""

                # Extract function name from tool_call_id or use generic name
                # OpenAI format doesn't include function name in tool response,
                # but Google needs it. We'll need to track this from previous messages.
                # For now, use a generic approach or extract from context
                func_name = msg.get("name", "unknown_function")

                # Create FunctionResponse part
                parts = [types.Part.from_function_response(
                    name=func_name,
                    response={"result": result_content}
                )]

                contents.append(types.Content(role="user", parts=parts))
                logger.info(f"✅ Converted tool result for '{func_name}' to Google function_response")

            else:
                logger.warning(f"⚠️ Skipping message with unknown role: {role}")

        # If no contents, add a default user message
        if not contents:
            logger.warning(f"⚠️ No valid contents after filtering, adding default message")
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello")]
            ))

        return contents, system_instruction

    def _clean_schema_for_google(self, schema: dict) -> dict:
        """Remove fields from JSON Schema that Google API doesn't support.

        Args:
            schema: JSON Schema dict (potentially with unsupported fields)

        Returns:
            Cleaned schema dict
        """
        if not isinstance(schema, dict):
            return schema

        # Fields to remove (not supported by Google)
        unsupported_fields = ["$schema", "additionalProperties"]

        cleaned = {}
        for key, value in schema.items():
            # Skip unsupported fields
            if key in unsupported_fields:
                continue

            # Recursively clean nested objects
            if isinstance(value, dict):
                cleaned[key] = self._clean_schema_for_google(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self._clean_schema_for_google(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned[key] = value

        return cleaned

    def _convert_tools_to_google_format(self, openai_tools: list) -> Optional[types.Tool]:
        """Convert OpenAI format tools to Google GenAI format.

        Args:
            openai_tools: List of OpenAI-format tool definitions

        Returns:
            Google types.Tool object or None
        """
        if not openai_tools:
            return None

        function_declarations = []

        for tool in openai_tools:
            # OpenAI format: {"type": "function", "function": {...}}
            if tool.get("type") == "function":
                func = tool.get("function", {})

                # Google format uses same schema structure for parameters
                function_decl = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                }

                # Clean and add parameters (remove unsupported fields like $schema)
                if "parameters" in func:
                    function_decl["parameters"] = self._clean_schema_for_google(func["parameters"])

                function_declarations.append(function_decl)
                logger.info(f"✅ Converted tool '{func.get('name')}' to Google format")

        if function_declarations:
            return types.Tool(function_declarations=function_declarations)

        return None

    def _build_config(self, request: Dict[str, Any], system_instruction: Optional[list]) -> Any:
        """Build GenerateContentConfig from request parameters.

        Args:
            request: OpenAI-format request dict
            system_instruction: Optional system instruction parts

        Returns:
            GenerateContentConfig object
        """
        config_params = {
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 0.95),
            "max_output_tokens": request.get("max_tokens", 8192),
            "safety_settings": [
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
        }

        # Convert and add tools if present
        if "tools" in request and request["tools"]:
            google_tools = self._convert_tools_to_google_format(request["tools"])
            if google_tools:
                config_params["tools"] = [google_tools]
                logger.info(f"✅ Added {len(request['tools'])} tools to config")

                # IMPORTANT: Disable automatic function calling so Claude Code handles execution
                # Google's automatic execution won't work because tool implementations are in Claude Code
                config_params["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
                logger.info("🔧 Disabled automatic function calling (Claude Code will execute tools)")

        # Add system instruction if present
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        return types.GenerateContentConfig(**config_params)

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
        # Extract text from Google response (handle both missing attribute and None value)
        text = getattr(response, "text", None) or ""

        # Check for empty response and log details
        if not text:
            logger.warning(f"⚠️ Google API returned empty response")
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None)
                safety_ratings = getattr(candidate, "safety_ratings", None)
                logger.warning(f"⚠️ Empty response - finish_reason={finish_reason}, safety_ratings={safety_ratings}")
            else:
                logger.warning(f"⚠️ Empty response - no candidates in response")

        # Build OpenAI-format response
        # Extract usage metadata (it's a Pydantic object, not a dict)
        usage_metadata = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        total_tokens = getattr(usage_metadata, "total_token_count", 0) if usage_metadata else 0

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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
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
        logger.info(f"🎬 Starting stream conversion for model {model}, request_id={request_id}")

        # Iterate through the stream and yield chunks in real-time
        # Use a queue to stream from blocking generator without waiting for all chunks
        import queue
        import threading

        chunk_queue = queue.Queue()
        error_holder = [None]
        chunk_count = [0]  # Track chunks for debugging

        def _stream_reader():
            """Read from blocking stream and put chunks in queue"""
            try:
                logger.info("📥 Stream reader thread started, reading from Google API...")
                for chunk in stream:
                    chunk_count[0] += 1
                    logger.info(f"📦 Stream reader: Got chunk #{chunk_count[0]}, queueing...")
                    chunk_queue.put(("chunk", chunk))
                logger.info(f"✅ Stream reader: Finished reading {chunk_count[0]} chunks, sending done signal")
                chunk_queue.put(("done", None))
            except Exception as e:
                logger.error(f"❌ Stream reader error: {str(e)}")
                logger.error(traceback.format_exc())
                error_holder[0] = e
                chunk_queue.put(("error", str(e)))

        # Start background thread to read stream
        reader_thread = threading.Thread(target=_stream_reader, daemon=True)
        reader_thread.start()
        logger.info("🔄 Background reader thread started")

        # Yield chunks as they arrive
        yielded_count = 0
        while True:
            # Get next item from queue with timeout to prevent infinite blocking
            logger.info(f"⏳ Main loop: Waiting for item from queue (yielded {yielded_count} so far)...")
            try:
                item_type, item_data = await asyncio.to_thread(chunk_queue.get, timeout=60.0)
                logger.info(f"✉️ Main loop: Got {item_type} from queue")
            except Exception as e:
                logger.error(f"❌ Queue.get() timeout or error: {str(e)}")
                break

            if item_type == "error":
                logger.error(f"❌ Error in stream: {item_data}")
                break
            elif item_type == "done":
                logger.info(f"🏁 Stream done, yielded {yielded_count} chunks total")
                break
            elif item_type != "chunk":
                logger.warning(f"⚠️ Unknown item type: {item_type}")
                continue

            chunk = item_data
            # Check for cancellation
            if (
                request_id
                and request_id in self.active_requests
                and self.active_requests[request_id].is_set()
            ):
                logger.info(f"🛑 Request {request_id} was cancelled, stopping stream")
                break

            # Extract text and function calls from chunk
            chunk_text = getattr(chunk, "text", None) or ""
            function_calls = []

            # Check for function_call parts in the chunk
            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            fc = part.function_call
                            function_calls.append({
                                "name": getattr(fc, "name", ""),
                                "args": dict(getattr(fc, "args", {}))
                            })

            # Log chunk details for debugging
            chunk_has_text = getattr(chunk, "text", None) is not None
            logger.info(f"📝 Chunk text length: {len(chunk_text)} chars (chunk.text={'present' if chunk_has_text else 'None'}), function_calls={len(function_calls)}")

            # Check for safety ratings or finish reasons if text is None
            if not chunk_has_text and not function_calls:
                # Log additional chunk attributes to understand why text is None
                chunk_attrs = dir(chunk)
                if hasattr(chunk, "candidates") and chunk.candidates:
                    candidate = chunk.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", None)
                    safety_ratings = getattr(candidate, "safety_ratings", None)
                    logger.warning(f"⚠️ Chunk with None text - finish_reason={finish_reason}, safety_ratings={safety_ratings}")
                else:
                    logger.warning(f"⚠️ Chunk with None text - chunk attributes: {[attr for attr in chunk_attrs if not attr.startswith('_')]}")

            # Build delta based on what's present
            delta = {}
            if chunk_text:
                delta["content"] = chunk_text

            if function_calls:
                # Convert Google function_call to OpenAI tool_calls format
                tool_calls = []
                for idx, fc in enumerate(function_calls):
                    tool_calls.append({
                        "id": f"call_{chunk_id}_{idx}",
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": json.dumps(fc["args"])
                        }
                    })
                delta["tool_calls"] = tool_calls
                logger.info(f"🔧 Converted {len(function_calls)} function calls to tool_calls")

            if delta:
                # Build OpenAI-format chunk
                chunk_data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                }

                # Yield as SSE event
                yielded_count += 1
                logger.info(f"⬆️ Yielding chunk #{yielded_count} to client...")
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                logger.info(f"✅ Chunk #{yielded_count} yielded successfully")

        # Check if we yielded any content
        if yielded_count == 0:
            logger.error(f"❌ Stream completed with 0 chunks yielded - model may have blocked the response")
            # Yield error event instead of normal completion
            error_data = {
                "error": {
                    "type": "content_filter_error",
                    "message": "The model did not generate any content. This may be due to safety filters or content policy restrictions."
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            logger.error(f"❌ Stream failed: No content generated")
        else:
            # Send final chunk with finish_reason
            logger.info("🎬 Sending final chunk with finish_reason=stop")
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
            logger.info(f"🏁 Stream complete. Total chunks yielded: {yielded_count}")

    async def cancel_request(self, request_id: str):
        """Cancel an ongoing request.

        Args:
            request_id: The request ID to cancel
        """
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            logger.info(f"Cancellation requested for {request_id}")
