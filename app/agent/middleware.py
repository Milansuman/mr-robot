import logging
from typing import Callable, Any
from datetime import datetime
from langchain.agents import AgentState
from langchain.agents.middleware import after_model, wrap_tool_call, wrap_model_call, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.runtime import Runtime
from agent.llm import FALLBACK_MODELS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PentestAgent")


# Logging Middleware
@after_model
def log_agent_thinking(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log the agent's response after each model call."""
    messages = state.get("messages", [])
    
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            # Log the agent's thinking
            if last_message.content:
                logger.info(f"🤖 Agent Thinking: {last_message.content[:200]}...")
            
            # Log tool calls if any
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    logger.info(f"🔧 Tool Invoked: {tool_name}")
                    logger.info(f"   Arguments: {tool_args}")
    
    return None


@wrap_tool_call  # type: ignore
def log_tool_execution(
    request: Any,
    handler: Callable[[Any], ToolMessage]
) -> ToolMessage:
    """Log tool execution and results."""
    tool_name = request.tool_call.get("name", "unknown") if hasattr(request, "tool_call") else "unknown"
    start_time = datetime.now()
    
    logger.info(f"⚙️  Executing Tool: {tool_name}")
    
    try:
        # Execute the tool
        result = handler(request)
        
        # Calculate execution time
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log the result - handle different return types
        if hasattr(result, "content"):
            result_preview = result.content[:150] if result.content else "No output"
            logger.info(f"✅ Tool Result ({duration:.2f}s): {result_preview}...")
        else:
            # Handle Command objects or other types without content
            logger.info(f"✅ Tool Executed ({duration:.2f}s): {tool_name} completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Tool Error: {tool_name} failed with {str(e)}")
        raise


@wrap_model_call
def rotate_models_on_rate_limit(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Try each model in sequence if any error occurs."""
    last_error = None
    
    for attempt, model in enumerate(FALLBACK_MODELS):
        try:
            logger.info(f"🎯 Using model: {model.model_name}")
            
            # Override the model in the request
            modified_request = request.override(model=model)
            
            # Execute the model call
            response = handler(modified_request)
            
            return response
            
        except Exception as e:
            error_str = str(e).lower()
            last_error = e
            
            # Determine error type for logging
            if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                error_type = "Rate limit"
            elif "timeout" in error_str:
                error_type = "Timeout"
            elif "connection" in error_str or "network" in error_str:
                error_type = "Connection"
            else:
                error_type = "Error"
            
            # Log and rotate to next model if available
            if attempt < len(FALLBACK_MODELS) - 1:
                logger.warning(
                    f"⚠️  {error_type} on {model.model_name}: {str(e)[:100]}. "
                    f"Rotating to next model..."
                )
            else:
                # If we've tried all models, raise error
                logger.error(
                    f"❌ All models exhausted. Last error: {error_type} - {str(e)[:100]}"
                )
                raise Exception(
                    f"All available models failed. Last error: {str(e)}"
                ) from last_error
    
    # Should never reach here, but just in case
    if last_error:
        raise last_error
    else:
        raise Exception("Failed to get response from any model")
