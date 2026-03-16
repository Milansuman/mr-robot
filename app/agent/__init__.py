from typing import Dict, Any, List, AsyncIterator
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import MemorySaver
from agent.llm import llm
from agent.tools import TOOLS
from agent.prompts import SYSTEM_PROMPT
from agent.middleware import (
    CacheMiddleware,
    LoggingMiddleware,
    ModelRotationMiddleware,
)
from langchain_core.runnables import RunnableConfig
from langchain_asynctools import AsyncTools

# Set up memory checkpointer for conversation persistence
checkpointer = MemorySaver()

# Create the penetration testing agent with all middleware
agent = create_agent(
    model=llm,
    tools=TOOLS,
    middleware=[  # type: ignore
        AsyncTools(),
        TodoListMiddleware(),
        # CacheMiddleware(),
        LoggingMiddleware(),
        ModelRotationMiddleware(),
    ],
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
)


async def invoke_agent(message: str, thread_id: str) -> Dict[str, Any]:
    """
    Invoke the penetration testing agent with a message and thread ID.
    
    Args:
        message: User message/command for the agent
        thread_id: Unique thread identifier for conversation context
        
    Returns:
        Dict with parsed response containing:
            - thread_id: The conversation thread ID
            - response: The agent's text response
            - tool_calls: List of tools called (if any)
    """
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config
    )
    
    # Extract messages from the result
    messages = result.get("messages", [])
    response_content = ""
    tool_calls: List[Dict[str, str]] = []
    
    # Get the last AI message as the response
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            response_content = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "assistant":
            response_content = msg.get("content", "")
            break
    
    # Extract tool calls if any
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            tool_calls.append({
                "tool": msg.name if hasattr(msg, "name") else "unknown",
                "status": "completed"
            })
    
    return {
        "thread_id": thread_id,
        "response": response_content or "Agent completed the task.",
        "tool_calls": tool_calls if tool_calls else None
    }


async def stream_agent(message: str, thread_id: str) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream the penetration testing agent's execution with real-time updates.
    
    Yields updates including tool calls, todo list changes, and final responses.
    
    Args:
        message: User message/command for the agent
        thread_id: Unique thread identifier for conversation context
        
    Yields:
        Dict with event type and data:
            - type: "tool_call" | "todo_update" | "thinking" | "response" | "error"
            - data: Event-specific data
    """
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    try:
        # Stream agent execution
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            config,
            stream_mode="updates"
        ):
            # Handle different types of events
            for node_name, node_output in chunk.items():
                # Skip if node_output is None
                if node_output is None:
                    continue
                
                # Tool execution events
                if isinstance(node_output, dict) and "messages" in node_output:
                    messages = node_output["messages"]
                    if messages is None:
                        continue
                    
                    for msg in messages:
                        if msg is None:
                            continue
                        
                        # Tool calls
                        if hasattr(msg, "type") and msg.type == "tool":
                            tool_name = msg.name if hasattr(msg, "name") else "unknown"
                            output_str = str(msg.content) if hasattr(msg, "content") and msg.content else ""
                            
                            yield {
                                "type": "tool_call",
                                "data": {
                                    "tool": tool_name,
                                    "status": "completed",
                                    "output": output_str[:500]  # Truncate for streaming
                                }
                            }
                            
                            # Special handling for write_todos tool to show planning updates
                            if tool_name == "write_todos":
                                yield {
                                    "type": "todo_update",
                                    "data": {
                                        "message": "Task plan updated",
                                        "todos": output_str
                                    }
                                }
                        # AI responses
                        elif hasattr(msg, "type") and msg.type == "ai":
                            if hasattr(msg, "content") and msg.content:
                                yield {
                                    "type": "response",
                                    "data": {
                                        "content": msg.content
                                    }
                                }
                
                # Todo list updates
                if isinstance(node_output, dict) and "todo_list" in node_output:
                    yield {
                        "type": "todo_update",
                        "data": {
                            "todos": node_output["todo_list"]
                        }
                    }
                
                # Thinking/agent reasoning
                if node_name == "agent" and isinstance(node_output, dict) and "messages" in node_output:
                    messages = node_output["messages"]
                    if messages is None:
                        continue
                    
                    # Extract thinking from AI messages
                    for msg in messages:
                        if msg is None:
                            continue
                        
                        if hasattr(msg, "type") and msg.type == "ai":
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    # Handle both dict and object tool_calls
                                    if isinstance(tool_call, dict):
                                        tool_name = tool_call.get("name", "tool")
                                        tool_args = tool_call.get("args", {})
                                    else:
                                        tool_name = getattr(tool_call, "name", "tool")
                                        tool_args = getattr(tool_call, "args", {})
                                    
                                    # Ensure args is a dict
                                    if not isinstance(tool_args, dict):
                                        tool_args = {}
                                    
                                    yield {
                                        "type": "thinking",
                                        "data": {
                                            "action": f"Calling {tool_name}",
                                            "args": tool_args
                                        }
                                    }
        
        # Send completion event
        yield {
            "type": "complete",
            "data": {
                "thread_id": thread_id,
                "status": "completed"
            }
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "data": {
                "error": str(e)
            }
        }


__all__ = ["agent", "invoke_agent", "stream_agent", "checkpointer"]

