"""
Search Tools for Task Agent
Hero Project: On-Device AI Agent with Vision and RAG
"""

import requests
from typing import Any
from atomic_agents.lib.tools.base_tool import BaseTool, BaseToolConfig

class WebSearchTool(BaseTool):
    """
    Tool for web search (mock implementation for local deployment)
    """
    
    def __init__(self):
        super().__init__(BaseToolConfig(
            name="web_search",
            description="Search the web for information (mock implementation for local deployment)",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ))
    
    def run(self, query: str, max_results: int = 3) -> str:
        """
        Perform web search (mock implementation)
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Mock search results
        """
        # Mock search results for local deployment
        mock_results = [
            {
                "title": f"Search Result 1 for '{query}'",
                "url": "https://example.com/result1",
                "snippet": f"This is a mock search result for the query '{query}'. In a real implementation, this would be actual web search results."
            },
            {
                "title": f"Search Result 2 for '{query}'",
                "url": "https://example.com/result2", 
                "snippet": f"Another mock result for '{query}'. This demonstrates how web search would work in the task agent."
            },
            {
                "title": f"Search Result 3 for '{query}'",
                "url": "https://example.com/result3",
                "snippet": f"Third mock result for '{query}'. For production use, integrate with a real search API like Google Custom Search or Bing Search."
            }
        ]
        
        results = mock_results[:max_results]
        
        formatted_results = "Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   URL: {result['url']}\n"
            formatted_results += f"   {result['snippet']}\n\n"
        
        formatted_results += "Note: This is a mock implementation for local deployment. For production use, integrate with a real search API."
        
        return formatted_results


