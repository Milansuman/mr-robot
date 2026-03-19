import logging
import re
from typing import Callable, Any, Awaitable
from datetime import datetime
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import ToolMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr
from langgraph.types import Command
from langgraph.runtime import Runtime
from agent.llm import FALLBACK_MODELS
from utils.cache import (
    generate_cache_key,
    get_from_cache,
    set_in_cache,
    is_cache_available
)
from config import env

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PentestAgent")


class CacheMiddleware(AgentMiddleware):
    """Caches model calls and deterministic tool calls to reduce redundant API requests."""

    @staticmethod
    def _tool_content_to_text(content: Any) -> str:
        """Normalize ToolMessage content into plain text for stable caching."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    @staticmethod
    def _is_async_placeholder(content_text: str) -> bool:
        """Detect async placeholder responses returned before a tool has completed."""
        return bool(
            re.search(
                r"tool call is being processed with job id:",
                content_text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _build_tool_message_from_cache(cached: Any, tool_call_id: str) -> ToolMessage | None:
        """Rebuild a ToolMessage from cached payload."""
        if isinstance(cached, dict):
            content = cached.get("content")
            if content is not None:
                return ToolMessage(content=str(content), tool_call_id=tool_call_id)
        elif isinstance(cached, str):
            return ToolMessage(content=cached, tool_call_id=tool_call_id)
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        if not is_cache_available():
            return await handler(request)

        try:
            messages_data = [
                {
                    "role": msg.type if hasattr(msg, "type") else "unknown",
                    "content": msg.content if hasattr(msg, "content") else str(msg)
                }
                for msg in request.state.get("messages", [])[-3:]
            ]
            model_name = getattr(request.model, "model_name", str(request.model))
            cache_data = {
                "model": model_name,
                "messages": messages_data,
                "system_prompt": getattr(request, "system_prompt", None),
            }
            cache_key = generate_cache_key("model_call", cache_data)

            cached_response = get_from_cache(cache_key)
            if cached_response:
                logger.info("💾 Cache hit for model call")
                return cached_response

            response = await handler(request)
            set_in_cache(cache_key, response, ttl=1800)
            return response

        except Exception as e:
            logger.debug(f"Model cache error: {e}")
            return await handler(request)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]]
    ) -> ToolMessage | Command[Any]:
        if not is_cache_available():
            return await handler(request)

        try:
            tool_name = request.tool_call.get("name", "") if hasattr(request, "tool_call") else ""
            tool_args = request.tool_call.get("args", {}) if hasattr(request, "tool_call") else {}

            # Only cache deterministic tools (not time-sensitive scans)
            cacheable_tools = {"send_http_request"}
            if tool_name not in cacheable_tools:
                return await handler(request)

            tool_call_id = (
                request.tool_call.get("id", "cached_tool_call")
                if hasattr(request, "tool_call") else "cached_tool_call"
            )
            cache_key = generate_cache_key("tool_call", {"tool": tool_name, "args": tool_args})
            cached_result = get_from_cache(cache_key)
            if cached_result:
                logger.info(f"💾 Cache hit for tool: {tool_name}")
                cached_tool_message = self._build_tool_message_from_cache(cached_result, tool_call_id)
                if cached_tool_message is not None:
                    return cached_tool_message

            result = await handler(request)
            if isinstance(result, ToolMessage):
                content_text = self._tool_content_to_text(result.content)

                # Skip async placeholder responses; cache only the real completed output.
                if self._is_async_placeholder(content_text):
                    logger.info(f"⏭️  Skipping placeholder cache for tool: {tool_name}")
                    return result

                logger.info(f"💾 Caching result for tool: {tool_name}")
                set_in_cache(
                    cache_key,
                    {"tool": tool_name, "content": content_text},
                    ttl=600,
                )
            return result

        except Exception as e:
            logger.debug(f"Tool cache error: {e}")
            return await handler(request)


class LoggingMiddleware(AgentMiddleware):
    """Logs agent thinking after each model call and tool execution details."""

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                if last_message.content:
                    logger.info(f"🤖 Agent Thinking: {last_message.content[:200]}...")
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        if tool_name == "write_todos":
                            logger.info("📝 Planning: Agent is creating/updating task list")
                            logger.info(f"   Tasks: {tool_args}")
                        else:
                            logger.info(f"🔧 Tool Invoked: {tool_name}")
                            logger.info(f"   Arguments: {tool_args}")
        return None

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]]
    ) -> ToolMessage | Command[Any]:
        tool_name = (
            request.tool_call.get("name", "unknown")
            if hasattr(request, "tool_call") else "unknown"
        )
        start_time = datetime.now()
        logger.info(f"⚙️  Executing Tool: {tool_name}")

        try:
            result = await handler(request)
            duration = (datetime.now() - start_time).total_seconds()
            if isinstance(result, ToolMessage) and result.content:
                result_preview = result.content[:150] if isinstance(result.content, str) else str(result.content)[:150]
                logger.info(f"✅ Tool Result ({duration:.2f}s): {result_preview}...")
            else:
                logger.info(
                    f"✅ Tool Executed ({duration:.2f}s): {tool_name} completed successfully"
                )
            return result
        except Exception as e:
            logger.error(f"❌ Tool Error: {tool_name} failed with {str(e)}")
            raise


class ModelRotationMiddleware(AgentMiddleware):
    """Rotates through fallback models and API keys on errors."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        last_error = None

        for attempt, model_name in enumerate(FALLBACK_MODELS):
            advance_to_next_model = False

            for key_index, api_key in enumerate(env.GROQ_API_KEYS_LIST):
                model = ChatGroq(
                    api_key=SecretStr(api_key),
                    model=model_name,
                    temperature=0.1,
                    max_retries=1,
                )

                max_retries = 1

                for retry in range(max_retries + 1):
                    try:
                        if retry == 0:
                            logger.info(
                                f"🎯 Using model: {model_name} "
                                f"with key {key_index + 1}/{len(env.GROQ_API_KEYS_LIST)}"
                            )
                        else:
                            logger.info(
                                f"🔄 Retry {retry}/{max_retries} for {model_name} "
                                f"(key {key_index + 1}/{len(env.GROQ_API_KEYS_LIST)})"
                            )

                        modified_request = request.override(model=model)
                        return await handler(modified_request)

                    except Exception as e:
                        error_str = str(e).lower()
                        error_repr = repr(e)
                        last_error = e

                        is_function_error = (
                            "failed to call a function" in error_str or
                            "adjust your prompt" in error_str or
                            "function call" in error_str or
                            (
                                "error" in error_repr and
                                "message" in error_repr and
                                "function" in error_str
                            )
                        )

                        if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                            error_type, should_retry = "Rate limit", False
                        elif "timeout" in error_str:
                            error_type, should_retry = "Timeout", False
                        elif "connection" in error_str or "network" in error_str:
                            error_type, should_retry = "Connection", False
                        elif is_function_error:
                            error_type, should_retry = "Function call error", False
                        else:
                            error_type, should_retry = "Error", False

                        if should_retry and retry < max_retries:
                            logger.warning(
                                f"⚠️  {error_type} on {model_name}: "
                                f"{str(e)[:100]}. Retrying..."
                            )
                            continue

                        if key_index < len(env.GROQ_API_KEYS_LIST) - 1:
                            logger.warning(
                                f"⚠️  {error_type} on {model_name}: "
                                f"{str(e)[:100]}. Rotating to next API key..."
                            )
                            break

                        if attempt < len(FALLBACK_MODELS) - 1:
                            logger.warning(
                                f"⚠️  {error_type} on {model_name}: "
                                f"{str(e)[:100]}. Rotating to next model..."
                            )
                            advance_to_next_model = True
                            break

                        logger.error(
                            f"❌ All models and keys exhausted. "
                            f"Last error: {error_type} - {str(e)[:100]}"
                        )
                        raise Exception(
                            f"All available models and keys failed. Last error: {str(e)}"
                        ) from last_error

                if advance_to_next_model:
                    break

        if last_error:
            raise last_error
        raise Exception("Failed to get response from any model")
