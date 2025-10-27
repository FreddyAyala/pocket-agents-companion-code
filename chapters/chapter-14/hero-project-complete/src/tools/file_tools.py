"""
File Tools for Task Agent
Hero Project: On-Device AI Agent with Vision and RAG
"""

import os
from typing import Any
from atomic_agents.lib.tools.base_tool import BaseTool, BaseToolConfig

class FileReadTool(BaseTool):
    """
    Tool for reading file contents
    """
    
    def __init__(self):
        super().__init__(BaseToolConfig(
            name="file_read",
            description="Read contents of a file from the local filesystem",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["file_path"]
            }
        ))
    
    def run(self, file_path: str) -> str:
        """
        Read file contents
        
        Args:
            file_path: Path to the file
            
        Returns:
            File contents as string
        """
        try:
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' does not exist"
            
            if not os.path.isfile(file_path):
                return f"Error: '{file_path}' is not a file"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return f"File contents of '{file_path}':\n\n{content}"
            
        except PermissionError:
            return f"Error: Permission denied reading '{file_path}'"
        except UnicodeDecodeError:
            return f"Error: Cannot decode file '{file_path}' as text"
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"

class FileWriteTool(BaseTool):
    """
    Tool for writing content to files
    """
    
    def __init__(self):
        super().__init__(BaseToolConfig(
            name="file_write",
            description="Write content to a file on the local filesystem",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path where to write the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        ))
    
    def run(self, file_path: str, content: str) -> str:
        """
        Write content to file
        
        Args:
            file_path: Path where to write the file
            content: Content to write
            
        Returns:
            Success or error message
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to '{file_path}'"
            
        except PermissionError:
            return f"Error: Permission denied writing to '{file_path}'"
        except Exception as e:
            return f"Error writing to file '{file_path}': {str(e)}"


