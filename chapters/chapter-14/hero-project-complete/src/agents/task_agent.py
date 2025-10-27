"""
Task Agent using Atomic Agents Framework
Hero Project: On-Device AI Agent with Vision and RAG
"""

from typing import Dict, Any, Optional, List
import logging
import os
import re

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class TaskAgent:
    """
    Task Automation Agent with REAL tool calling capabilities
    """
    
    def __init__(self, model_loader, system_prompt: Optional[str] = None):
        """
        Initialize Task Agent
        
        Args:
            model_loader: Qwen3VLLoader instance
            system_prompt: Custom system prompt (optional)
        """
        self.model_loader = model_loader
        
        # Initialize tools
        self.tools = {
            "file_read": self._file_read,
            "file_write": self._file_write,
            "web_search": self._web_search
        }
        
        # Default system prompt - REAL AGENTIC APPROACH
        default_system_prompt = """You are an autonomous AI agent with access to various tools.
        You must decide for yourself when and how to use tools to accomplish tasks.
        
        Available tools:
        - file_read(path): Read contents of files
        - file_write(path, content): Write content to files  
        - web_search(query): Search the web for information
        
        TOOL CALLING FORMAT:
        When you need to use a tool, respond with exactly this format:
        TOOL: tool_name(parameter1, parameter2)
        
        Examples:
        - TOOL: file_read(test.txt)
        - TOOL: file_write(summary.txt, This is the content to write)
        - TOOL: web_search(latest AI news)
        
        AGENTIC BEHAVIOR RULES:
        1. AUTONOMOUS DECISION MAKING: Decide for yourself which tools to use
        2. USE TOOL FORMAT: Always use "TOOL: tool_name(params)" when calling tools
        3. SEARCH WHEN UNSURE: If you don't know something, use web_search
        4. READ FILES WHEN ASKED: If asked about file contents, use file_read
        5. SAVE INFORMATION: If asked to save/record info, use file_write
        6. MULTI-STEP TASKS: Use multiple tools in sequence when needed
        
        Be proactive and autonomous in your tool usage!"""
        
        self.system_prompt = system_prompt or default_system_prompt
        print("✅ Task Agent initialized")
    
    def _file_read(self, file_path: str) -> str:
        """Read file contents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File '{file_path}' contents:\n{content}"
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"
    
    def _file_write(self, file_path: str, content: str) -> str:
        """Write content to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote content to '{file_path}'"
        except Exception as e:
            return f"Error writing to file '{file_path}': {str(e)}"
    
    def _web_search(self, query: str) -> str:
        """Mock web search"""
        return f"Mock search results for '{query}': This is a demonstration of web search functionality. In a real implementation, this would connect to a search API."
    
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from LLM response"""
        # Look for TOOL: tool_name(params) pattern
        tool_pattern = r'TOOL:\s*(\w+)\((.*?)\)'
        match = re.search(tool_pattern, response)
        
        if match:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters (simple comma-separated)
            if params_str.strip():
                params = [p.strip().strip('"\'') for p in params_str.split(',')]
            else:
                params = []
            
            return {
                "name": tool_name,
                "params": params
            }
        
        return None
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call"""
        tool_name = tool_call["name"]
        params = tool_call["params"]
        
        if tool_name in self.tools:
            try:
                if tool_name == "file_read":
                    return self.tools[tool_name](params[0])
                elif tool_name == "file_write":
                    return self.tools[tool_name](params[0], params[1])
                elif tool_name == "web_search":
                    return self.tools[tool_name](params[0])
                else:
                    return f"Unknown tool: {tool_name}"
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        else:
            return f"Tool '{tool_name}' not found"
    
    def run(self, task: str, image=None) -> Dict[str, Any]:
        """
        Run task with REAL autonomous tool calling
        
        Args:
            task: Task description
            image: Optional PIL image
            
        Returns:
            Dictionary with result and tools used
        """
        try:
            # Build messages for the model
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task}
            ]
            
            # Generate response using the model
            print(f"🤖 Processing task: '{task}'")
            response = self.model_loader.generate_response(messages, images=[image] if image else None)
            
            # Check if response contains tool calls
            tool_call = self._parse_tool_call(response)
            tools_used = []
            
            if tool_call:
                print(f"🔧 Agent decided to use tool: {tool_call['name']}")
                tool_result = self._execute_tool(tool_call)
                tools_used.append(tool_call['name'])
                
                # Get final response with tool result
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result: {tool_result}. Now provide the final answer."})
                
                final_response = self.model_loader.generate_response(messages, images=[image] if image else None)
                
                return {
                    "result": final_response,
                    "tools_used": tools_used,
                    "tool_result": tool_result,
                    "task": task
                }
            else:
                # No tool call detected, return direct response
                return {
                    "result": response,
                    "tools_used": [],
                    "task": task
                }
            
        except Exception as e:
            return {
                "result": f"Error processing task: {str(e)}",
                "tools_used": [],
                "task": task
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "type": "Task Agent",
            "model": self.model_loader.model_path,
            "tools": list(self.tools.keys()),
            "capabilities": ["file_operations", "web_search", "task_automation"]
        }