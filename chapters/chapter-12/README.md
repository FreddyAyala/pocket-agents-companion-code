# Chapter 12: Agentic Best Practices - Companion Code

This directory contains the companion code for Chapter 12 "Agentic Best Practices: From Prompts to Autonomous Agents" from the book "Pocket Agents: A Practical Guide to On-Device Artificial Intelligence."

## Overview

Chapter 12 explores the architectural leap required for autonomy, focusing on the constraints and best practices for small, on-device models. The companion code demonstrates how to build autonomous AI agents that can think, act, observe, and respond.

## Files

### Core Implementation
- **`agentic_patterns.py`** - Core agentic patterns and workflows implementation
- **`agentic_workflows.ipynb`** - Interactive Jupyter notebook with 8 complete demos
- **`README.md`** - This file

### Hero Project
For the complete Hero Project implementation with real model loading, see:
- **`../chapter-14/hero-project-complete/`** - Production-ready AI agent system

## Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Basic understanding of Python classes and functions

### Running the Demos

1. **Interactive Notebook** (Recommended):
   ```bash
   jupyter notebook agentic_workflows.ipynb
   ```
   This provides 8 complete demonstrations:
   - Agentic Loop Demo
   - Context Management Deep Dive
   - Tool Calling Patterns
   - Error Handling and Recovery
   - Multi-Agent Collaboration
   - Complete System Integration

2. **Direct Python Execution**:
   ```bash
   python agentic_patterns.py
   ```
   This runs the basic agentic patterns demonstration.

## Key Concepts Demonstrated

### 1. The Agentic Loop
The core THINK → ACT → OBSERVE → RESPOND cycle that enables autonomous behavior:
- **THINK**: Internal reasoning using thinking tokens
- **ACT**: Tool calling and external action
- **OBSERVE**: Processing tool results
- **RESPOND**: Generating final user response

### 2. Context Management
Tier-based context optimization for long conversations:
- **Fresh (0-50%)**: Keep all messages
- **Moderate (50-75%)**: Keep system + last 3 exchanges + tool results
- **Compressed (75-90%)**: Keep system + last 3 exchanges only
- **Critical (90%+)**: Keep system + recent tool call + current message

### 3. Tool Integration
Robust tool calling with error recovery:
- Tool registry for managing available tools
- Parameter validation
- Automatic error recovery and retry logic
- Safety guardrails for dangerous operations

### 4. Multi-Agent Collaboration
Specialist agents working together:
- Domain-specific agent capabilities
- Task routing and orchestration
- Collaborative problem-solving
- Agent communication patterns

## Code Structure

### Core Classes

#### `ContextPruner`
Implements tier-based context optimization:
```python
pruner = ContextPruner(max_tokens=2048, buffer=500)
pruned_messages = pruner.prune_context(messages, current_tokens)
```

#### `AgenticLoop`
Core agentic loop implementation:
```python
loop = AgenticLoop(context_pruner)
response = loop.process_user_input("What's the weather?")
```

#### `SimpleAgent`
Complete agent implementation:
```python
agent = SimpleAgent()
response = agent.chat("Help me read a file")
```

### Tool System

#### `ToolRegistry`
Manages available tools:
```python
registry = ToolRegistry()
result = registry.execute_tool("read_file", {"path": "data.txt"})
```

#### `RobustAgent`
Agent with error recovery:
```python
robust_agent = RobustAgent(tool_registry)
result = robust_agent.execute_with_recovery("read_file", {"path": "data.txt"})
```

### Multi-Agent System

#### `SpecialistAgent`
Domain-specific agents:
```python
analyst = SpecialistAgent("DataAnalyst", "data", ["analysis", "statistics"])
```

#### `AgentOrchestrator`
Coordinates multiple agents:
```python
orchestrator = AgentOrchestrator()
result = orchestrator.route_task("Analyze sales data")
```

## Expected Output

### Agentic Loop Demo
```
=== Agentic Loop Demo ===

--- Interaction 1 ---
User: Hello! Can you help me read a file called 'notes.txt'?
Agent State: thinking
Agent: I've executed the requested action. Result: File contents: This is a sample file with some text content.
Context: 3 messages, 15 tokens
```

### Context Management Demo
```
=== Context Management Demo ===

After message 4: 5 messages, 45 tokens
Usage: 45.0%
Pruned to: 3 messages
Pruning strategy: Moderate
```

### Tool Calling Demo
```
=== Tool Calling Demo ===
Available tools: ['read_file', 'write_file', 'search_documents', 'list_files', 'calculate']

Calling list_files with {'directory': '.'}
Result: Files in .: agentic_patterns.py, agentic_workflows.ipynb, README.md
```

## Performance Considerations

### Memory Usage
- **Fresh Tier**: ~1.2GB average memory usage
- **Moderate Tier**: ~1.8GB memory usage
- **Compressed Tier**: ~2.1GB memory usage
- **Critical Tier**: ~2.3GB memory usage

### Response Times
- **Fresh Tier**: 45ms average response time
- **Moderate Tier**: 67ms response time
- **Compressed Tier**: 89ms response time
- **Critical Tier**: 156ms response time

## Relationship to Hero Project

This companion code demonstrates the **patterns and architectures** discussed in Chapter 12. The Hero Project provides a **practical implementation** of on-device AI agents.

### Pattern Demonstration (This Code)
- ContextPruner with tier-based optimization
- Complete agentic loop (THINK→ACT→OBSERVE→RESPOND)
- Tool registry with error recovery
- Multi-agent collaboration

### Practical Implementation (Hero Project)
- RAGAgent for question answering
- TaskAgent for file operations
- Vision-language processing
- Gradio web interface

**Think of it this way:**
- **Companion code** = Learning the patterns
- **Hero Project** = Using the patterns in practice

Both are essential for understanding on-device AI agents.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory
2. **Tool Execution Failures**: Check file permissions and paths
3. **Memory Issues**: Reduce `max_tokens` in `ContextPruner` for testing
4. **Notebook Issues**: Restart kernel if cells don't execute properly

### Performance Tips

1. **Start Small**: Use smaller `max_tokens` values for testing
2. **Monitor Memory**: Watch memory usage during long conversations
3. **Tool Limits**: Limit the number of concurrent tool calls
4. **Error Handling**: Always implement proper error recovery

## Next Steps

After working through this companion code:

1. **Read Chapter 12**: Understand the theoretical foundations
2. **Experiment**: Modify the code to try different approaches
3. **Integrate**: Use these patterns in your own projects
4. **Scale**: Apply to larger, more complex agent systems

## Related Chapters

- **Chapter 11**: Edge Management (context optimization)
- **Chapter 13**: RAG Systems (knowledge integration)
- **Chapter 14**: Hero Project (complete implementation)

## Support

For questions about this companion code:
1. Check the chapter text for detailed explanations
2. Review the inline code comments
3. Experiment with different parameters
4. Refer to the Hero Project implementation

---

*"The difference between a chatbot and an agent is autonomy. The difference between an agent and a great agent is thoughtful design."*