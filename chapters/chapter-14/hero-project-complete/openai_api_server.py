#!/usr/bin/env python3
"""
OpenAI-Compatible API Server for Atomic Agents
Hero Project: On-Device AI Agent with Vision and RAG

This server exposes our Atomic Agents as an OpenAI-compatible API,
allowing integration with any UI that supports OpenAI endpoints:
- HuggingFace Chat-UI
- oobabooga text-generation-webui  
- SillyTavern
- LM Studio
- And many more!
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add src to path
sys.path.append('src')

# FastAPI for the API server
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Our Atomic Agents
from model_loader import Qwen3VLLoader
from vector_store import VectorStore
from agents.rag_agent import RAGAgent
from agents.task_agent import TaskAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for our agents
model_loader = None
vector_store = None
rag_agent = None
task_agent = None
system_initialized = False

# Available tools for OpenAI compatibility
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file from the local filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "file_write",
            "description": "Write content to a file on the local filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path where to write the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# FastAPI app
app = FastAPI(
    title="Hero Project OpenAI API",
    description="OpenAI-compatible API for Atomic Agents with RAG and Tool Calling",
    version="1.0.0"
)

# CORS middleware for web UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ChatCompletionRequest(BaseModel):
    model: str = "hero-project-agent"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "hero-project"

# Tool execution functions
def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> str:
    """Execute a tool and return the result"""
    try:
        if tool_name == "file_read":
            from tools.file_tools import FileReadTool
            tool = FileReadTool()
            return tool.call(parameters["path"])
            
        elif tool_name == "file_write":
            from tools.file_tools import FileWriteTool
            tool = FileWriteTool()
            return tool.call(parameters["path"], parameters["content"])
            
        elif tool_name == "web_search":
            from tools.search_tools import WebSearchTool
            tool = WebSearchTool()
            return tool.call(parameters["query"])
            
        else:
            return f"Unknown tool: {tool_name}"
            
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

def parse_agent_response(response: str) -> tuple[str, List[Dict]]:
    """Parse our agent's response and extract tool calls if any"""
    import re
    
    # Look for our custom tool format: TOOL: tool_name(params)
    tool_pattern = r'TOOL:\s*(\w+)\((.*?)\)'
    tool_calls = []
    
    # Find all tool calls in the response
    matches = re.findall(tool_pattern, response)
    
    for i, (tool_name, params_str) in enumerate(matches):
        # Parse parameters (simple parsing for now)
        params = {}
        if params_str.strip():
            # Handle simple parameter parsing
            if ',' in params_str:
                parts = params_str.split(',')
                if len(parts) == 2:
                    params["path"] = parts[0].strip().strip('"\'')
                    params["content"] = parts[1].strip().strip('"\'')
                else:
                    params["query"] = params_str.strip().strip('"\'')
            else:
                params["path"] = params_str.strip().strip('"\'')
        
        tool_calls.append({
            "id": f"call_{i+1}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(params)
            }
        })
    
    # Remove tool calls from the response text
    clean_response = re.sub(tool_pattern, '', response).strip()
    
    return clean_response, tool_calls

# Initialize system
async def initialize_system():
    """Initialize the Atomic Agents system"""
    global model_loader, vector_store, rag_agent, task_agent, system_initialized
    
    try:
        logger.info("🚀 Initializing Hero Project Atomic Agents...")
        
        # Initialize model loader
        logger.info("📦 Loading Qwen3-4B model...")
        model_loader = Qwen3VLLoader("unsloth/Qwen3-4B-Instruct-2507-GGUF")
        model_loader.load()
        
        # Initialize vector store
        logger.info("🗄️ Setting up vector store...")
        vector_store = VectorStore()
        vector_store.save_sample_documents()
        
        # Initialize agents
        logger.info("🤖 Creating RAG Agent...")
        rag_agent = RAGAgent(model_loader, vector_store)
        
        logger.info("🛠️ Creating Task Agent...")
        task_agent = TaskAgent(model_loader)
        
        system_initialized = True
        logger.info("✅ Hero Project system initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise

# OpenAI-compatible endpoints
@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "hero-project-agent",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "hero-project"
            }
        ]
    }

@app.get("/v1/tools")
async def list_tools():
    """List available tools (OpenAI compatible)"""
    return {
        "object": "list",
        "data": AVAILABLE_TOOLS
    }

@app.post("/v1/tools/execute")
async def execute_tool_call(request: Dict[str, Any]):
    """Execute a tool call (for UI integration)"""
    try:
        tool_name = request.get("name")
        parameters = request.get("parameters", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        
        result = execute_tool(tool_name, parameters)
        
        return {
            "tool_call_id": request.get("id", "unknown"),
            "result": result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Error executing tool: {e}")
        return {
            "tool_call_id": request.get("id", "unknown"),
            "result": f"Error: {str(e)}",
            "success": False
        }

async def generate_streaming_response(request: ChatCompletionRequest):
    """Generate streaming response for Chat-UI"""
    global system_initialized
    
    if not system_initialized:
        await initialize_system()
    
    try:
        # Convert messages to conversation format
        conversation = []
        for msg in request.messages:
            conversation.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Get the last user message
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            yield f"data: {json.dumps({'error': 'No user message found'})}\n\n"
            return
        
        # Generate response - auto-detect tool usage
        should_use_tools = request.tools or any(keyword in user_message.lower() for keyword in [
            'read', 'file', 'write', 'search', 'web', 'look up', 'find', 'get', 'show me'
        ])
        
        if should_use_tools:
            logger.info("🛠️ Using Task Agent for tool calling (streaming, auto-detected)")
            result = task_agent.run(user_message)
            response_content = result.get("result", "Task completed.")
        else:
            logger.info("💬 Using general assistant mode (streaming)")
            response_content = model_loader.generate_response(conversation)
        
        # Parse response for tool calls
        clean_content, tool_calls = parse_agent_response(response_content)
        final_content = clean_content if clean_content else response_content
        
        # Stream the response
        words = final_content.split()
        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.05)  # Small delay for streaming effect
        
        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"❌ Error in streaming response: {e}")
        error_chunk = {
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion (OpenAI compatible)"""
    global system_initialized
    
    logger.info(f"📨 Received chat completion request: {request.model}")
    logger.info(f"📝 Messages: {len(request.messages)} messages")
    logger.info(f"🔧 Tools requested: {bool(request.tools)}")
    logger.info(f"🌊 Streaming requested: {getattr(request, 'stream', False)}")
    
    if not system_initialized:
        await initialize_system()
    
    # Check if streaming is requested
    if getattr(request, 'stream', False):
        return StreamingResponse(
            generate_streaming_response(request),
            media_type="text/plain"
        )
    
    try:
        # Convert messages to conversation format
        conversation = []
        for msg in request.messages:
            conversation.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Get the last user message
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Check if tools are requested OR if user is asking for file operations
        should_use_tools = request.tools or any(keyword in user_message.lower() for keyword in [
            'read', 'file', 'write', 'search', 'web', 'look up', 'find', 'get', 'show me'
        ])
        
        if should_use_tools:
            logger.info("🛠️ Using Task Agent for tool calling (auto-detected)")
            result = task_agent.run(user_message)
            response_content = result.get("result", "Task completed.")
        else:
            # Use general assistant mode
            logger.info("💬 Using general assistant mode")
            response_content = model_loader.generate_response(conversation)
        
        # Parse response for tool calls
        clean_content, tool_calls = parse_agent_response(response_content)
        
        # Create message object
        message = {
            "role": "assistant",
            "content": clean_content if clean_content else response_content
        }
        
        # Add tool calls if any
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        # Create OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ],
            usage={
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(clean_content.split()) if clean_content else len(response_content.split()),
                "total_tokens": len(user_message.split()) + (len(clean_content.split()) if clean_content else len(response_content.split()))
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_initialized": system_initialized,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Hero Project OpenAI-Compatible API",
        "description": "Atomic Agents with RAG and Tool Calling",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "health": "/health"
        },
        "supported_uis": [
            "HuggingFace Chat-UI",
            "oobabooga text-generation-webui",
            "SillyTavern",
            "LM Studio",
            "Ollama WebUI",
            "And any OpenAI-compatible client!"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Hero Project OpenAI API Server...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("")
    print("🔗 Compatible with:")
    print("   • HuggingFace Chat-UI")
    print("   • oobabooga text-generation-webui")
    print("   • SillyTavern")
    print("   • LM Studio")
    print("   • Any OpenAI-compatible client!")
    print("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
