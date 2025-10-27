"""
Simple Agent Implementation
Chapter 14: Agentic Best Practices
Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence

This module provides a minimal but complete agent implementation demonstrating
core agentic AI patterns including tool calling, state management, and autonomous decision-making.
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"

@dataclass
class Tool:
    """Represents a tool that the agent can use"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        try:
            result = self.function(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

@dataclass
class AgentMemory:
    """Agent memory system for maintaining context"""
    short_term: List[Dict[str, Any]] = None
    long_term: Dict[str, Any] = None
    max_short_term: int = 10
    
    def __post_init__(self):
        """Initialize default values"""
        if self.short_term is None:
            self.short_term = []
        if self.long_term is None:
            self.long_term = {}
    
    def add_interaction(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add an interaction to short-term memory"""
        interaction = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.short_term.append(interaction)
        
        # Maintain max short-term memory size
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)
    
    def get_context(self) -> List[Dict[str, Any]]:
        """Get recent context for the agent"""
        return self.short_term[-5:]  # Last 5 interactions
    
    def store_long_term(self, key: str, value: Any):
        """Store information in long-term memory"""
        self.long_term[key] = value

class SimpleAgent:
    """
    A simple but complete agent implementation demonstrating agentic AI patterns
    """
    
    def __init__(self, name: str = "Agent", system_prompt: str = None):
        """
        Initialize the agent
        
        Args:
            name: Agent name
            system_prompt: System prompt for the agent
        """
        self.name = name
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.tools = {}
        
        # Default system prompt
        self.system_prompt = system_prompt or f"""You are {name}, an autonomous AI agent. 
        Your role is to help users accomplish tasks by thinking through problems, 
        using available tools, and providing helpful responses.
        
        Guidelines:
        - Think step by step before acting
        - Use tools when they would be helpful
        - Be clear about your reasoning
        - Ask for clarification when needed
        - Learn from your interactions"""
        
        logger.info(f"Agent {name} initialized")
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent's toolkit"""
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")
    
    def get_available_tools(self) -> str:
        """Get description of available tools"""
        if not self.tools:
            return "No tools available"
        
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        return "Available tools:\n" + "\n".join(tool_descriptions)
    
    def think(self, user_input: str) -> str:
        """
        Agent thinking process - analyze the situation and plan actions
        
        Args:
            user_input: User's input/request
            
        Returns:
            Agent's thinking/reasoning
        """
        self.state = AgentState.THINKING
        
        # Get context from memory
        context = self.memory.get_context()
        
        # Simple thinking logic (in a real implementation, this would use an LLM)
        thinking = f"""
        Analyzing request: "{user_input}"
        
        Context from recent interactions:
        {self._format_context(context)}
        
        Available tools: {list(self.tools.keys())}
        
        Planning next steps:
        1. Understand the user's request
        2. Determine if tools are needed
        3. Execute appropriate actions
        4. Provide helpful response
        """
        
        # Store thinking in memory
        self.memory.add_interaction("agent", f"Thinking: {thinking}")
        
        logger.info(f"Agent thinking: {thinking[:100]}...")
        return thinking
    
    def act(self, user_input: str) -> Dict[str, Any]:
        """
        Agent action process - execute planned actions
        
        Args:
            user_input: User's input/request
            
        Returns:
            Action results
        """
        self.state = AgentState.ACTING
        
        # Simple action logic (in a real implementation, this would use an LLM)
        # For demo purposes, we'll simulate tool usage based on keywords
        
        action_results = []
        
        # Check if any tools should be used
        for tool_name, tool in self.tools.items():
            if self._should_use_tool(user_input, tool_name):
                logger.info(f"Using tool: {tool_name}")
                
                # Execute tool with appropriate parameters
                tool_result = tool.execute()
                action_results.append({
                    "tool": tool_name,
                    "result": tool_result
                })
        
        # Store action in memory
        self.memory.add_interaction("agent", f"Actions taken: {action_results}")
        
        return {
            "actions": action_results,
            "status": "completed"
        }
    
    def respond(self, user_input: str) -> str:
        """
        Complete agent response cycle
        
        Args:
            user_input: User's input/request
            
        Returns:
            Agent's response
        """
        # Store user input in memory
        self.memory.add_interaction("user", user_input)
        
        # Think through the problem
        thinking = self.think(user_input)
        
        # Take actions
        actions = self.act(user_input)
        
        # Generate response
        response = self._generate_response(user_input, thinking, actions)
        
        # Store response in memory
        self.memory.add_interaction("agent", response)
        
        # Update state
        self.state = AgentState.IDLE
        
        return response
    
    def _should_use_tool(self, user_input: str, tool_name: str) -> bool:
        """Determine if a tool should be used based on user input"""
        # Simple keyword-based tool selection
        tool_keywords = {
            "calculator": ["calculate", "math", "compute", "add", "subtract", "multiply", "divide"],
            "file_manager": ["file", "save", "read", "write", "create", "delete"],
            "web_search": ["search", "find", "look up", "information", "news"],
            "weather": ["weather", "temperature", "forecast", "rain", "sunny"]
        }
        
        if tool_name in tool_keywords:
            keywords = tool_keywords[tool_name]
            return any(keyword in user_input.lower() for keyword in keywords)
        
        return False
    
    def _generate_response(self, user_input: str, thinking: str, actions: Dict[str, Any]) -> str:
        """Generate final response to user"""
        response_parts = []
        
        # Acknowledge the request
        response_parts.append(f"I understand you want help with: {user_input}")
        
        # Include thinking if relevant
        if "complex" in user_input.lower() or "difficult" in user_input.lower():
            response_parts.append(f"\nLet me think through this step by step...")
        
        # Include action results
        if actions["actions"]:
            response_parts.append(f"\nI've taken the following actions:")
            for action in actions["actions"]:
                tool_name = action["tool"]
                result = action["result"]
                if result["success"]:
                    response_parts.append(f"- Used {tool_name}: {result['result']}")
                else:
                    response_parts.append(f"- {tool_name} failed: {result['error']}")
        
        # Provide helpful response
        response_parts.append(f"\nIs there anything else I can help you with?")
        
        return "\n".join(response_parts)
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for display"""
        if not context:
            return "No previous context"
        
        formatted = []
        for interaction in context:
            role = interaction["role"]
            content = interaction["content"][:100] + "..." if len(interaction["content"]) > 100 else interaction["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "tools_available": list(self.tools.keys()),
            "memory_size": len(self.memory.short_term),
            "long_term_keys": list(self.memory.long_term.keys())
        }

# Example tool implementations
def calculator_tool(operation: str = "add", a: float = 0, b: float = 0) -> str:
    """Simple calculator tool"""
    if operation == "add":
        return f"{a} + {b} = {a + b}"
    elif operation == "subtract":
        return f"{a} - {b} = {a - b}"
    elif operation == "multiply":
        return f"{a} * {b} = {a * b}"
    elif operation == "divide":
        return f"{a} / {b} = {a / b}" if b != 0 else "Error: Division by zero"
    else:
        return f"Unknown operation: {operation}"

def file_manager_tool(action: str = "list", filename: str = "") -> str:
    """Simple file manager tool"""
    if action == "list":
        return "Files in current directory: example.txt, data.json, config.yaml"
    elif action == "read":
        return f"Reading file {filename}: [File content would be here]"
    elif action == "create":
        return f"Created file {filename}"
    else:
        return f"Unknown action: {action}"

def web_search_tool(query: str = "") -> str:
    """Simple web search tool"""
    return f"Search results for '{query}': [Search results would be here]"

def weather_tool(location: str = "current") -> str:
    """Simple weather tool"""
    return f"Weather for {location}: 72°F, partly cloudy"

# Demo function
def demo_simple_agent():
    """Demonstrate the simple agent implementation"""
    print("🤖 Simple Agent Demo")
    print("=" * 25)
    
    # Create agent
    agent = SimpleAgent("Assistant")
    
    # Add tools
    agent.add_tool(Tool(
        name="calculator",
        description="Perform mathematical calculations",
        parameters={"operation": "str", "a": "float", "b": "float"},
        function=calculator_tool
    ))
    
    agent.add_tool(Tool(
        name="file_manager",
        description="Manage files and directories",
        parameters={"action": "str", "filename": "str"},
        function=file_manager_tool
    ))
    
    agent.add_tool(Tool(
        name="web_search",
        description="Search the web for information",
        parameters={"query": "str"},
        function=web_search_tool
    ))
    
    agent.add_tool(Tool(
        name="weather",
        description="Get weather information",
        parameters={"location": "str"},
        function=weather_tool
    ))
    
    # Test interactions
    test_inputs = [
        "Hello, can you help me calculate 15 + 27?",
        "What's the weather like today?",
        "Can you search for information about AI?",
        "List the files in my directory"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        response = agent.respond(user_input)
        print(f"🤖 Agent: {response}")
        print("-" * 50)
    
    # Show agent status
    status = agent.get_status()
    print(f"\n📊 Agent Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    demo_simple_agent()
