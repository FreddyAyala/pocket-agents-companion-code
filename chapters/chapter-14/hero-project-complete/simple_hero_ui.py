#!/usr/bin/env python3
"""
Simple Hero Project Chat Interface
Using official Gradio ChatInterface for clean, working UI
"""

import gradio as gr
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from model_loader import Qwen3VLLoader
    from vector_store import VectorStore
    from agents.rag_agent import RAGAgent
    from agents.task_agent import TaskAgent
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class SimpleHeroInterface:
    def __init__(self):
        self.model_loader = None
        self.vector_store = None
        self.rag_agent = None
        self.task_agent = None
        self.initialized = False
        
    def initialize_system(self):
        """Initialize the AI system components"""
        if self.initialized:
            return "System already initialized!"
            
        try:
            print("🔄 Initializing Hero Project AI...")
            
            # Initialize model loader
            print("📥 Loading Qwen3-4B model...")
            self.model_loader = Qwen3VLLoader("unsloth/Qwen3-4B-Instruct-2507-GGUF")
            self.model_loader.load()
            
            # Initialize vector store
            print("🗄️ Setting up vector database...")
            self.vector_store = VectorStore()
            self.vector_store.save_sample_documents()
            
            # Initialize agents
            print("🤖 Creating AI agents...")
            self.rag_agent = RAGAgent(self.model_loader, self.vector_store)
            self.task_agent = TaskAgent(self.model_loader)
            
            self.initialized = True
            print("✅ Hero Project AI initialized successfully!")
            return "✅ System ready! Ask me anything."
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return f"❌ Initialization failed: {e}"

    def _format_capabilities_response(self, response):
        """Format capabilities response with better structure"""
        if "Mock response" in response:
            return """## 🛠️ **Hero Project AI Capabilities**

I'm a local AI assistant powered by **Qwen3-4B-Instruct** with **Atomic Agents** framework. Here's what I can help you with:

### 🔧 **Core Features:**
- **🤖 General Chat**: Natural conversations and Q&A
- **📚 RAG Search**: Search through knowledge base for information
- **🛠️ Tool Execution**: Read/write files, web search, and more

### 💡 **What I Can Do:**
1. **📖 File Operations**: Read and write files on your system
2. **🔍 Knowledge Search**: Find information from my knowledge base
3. **💬 General Assistance**: Answer questions and have conversations
4. **🧠 Problem Solving**: Help with logical reasoning and analysis
5. **📝 Writing Help**: Assist with content creation and editing

### 🚀 **Try These Examples:**
- "Read the README.md file"
- "What do you know about machine learning?"
- "Write a test file called hello.txt"
- "Search for information about AI"

**Note**: This is a demonstration with mock responses. In a full implementation, I would use the actual Qwen3-4B model for real AI capabilities!"""
        return response

    def chat_with_agent(self, message, history):
        """Main chat function for Gradio ChatInterface with streaming and thinking tokens"""
        if not self.initialized:
            self.initialize_system()
        
        # Auto-detect if user wants to use tools
        tool_keywords = ['read', 'file', 'write', 'search', 'web', 'look up', 'find', 'get', 'show me']
        use_tools = any(keyword in message.lower() for keyword in tool_keywords)
        
        try:
            # Show thinking process
            thinking_steps = []
            
            if use_tools:
                thinking_steps = [
                    "🤔 **Analyzing your request...**",
                    "🛠️ **Detected tool usage - preparing Task Agent...**",
                    "⚡ **Executing tools...**",
                    "✅ **Processing results...**"
                ]
            elif any(keyword in message.lower() for keyword in ['knowledge', 'learn', 'about', 'what is', 'explain']):
                thinking_steps = [
                    "🤔 **Analyzing your question...**",
                    "🔍 **Searching knowledge base...**",
                    "📚 **Retrieving relevant information...**",
                    "💡 **Synthesizing answer...**"
                ]
            else:
                thinking_steps = [
                    "🤔 **Thinking about your message...**",
                    "🧠 **Processing with general assistant...**",
                    "💭 **Formulating response...**"
                ]
            
            # Stream thinking tokens progressively
            current_response = ""
            for step in thinking_steps:
                current_response += step + "\n"
                time.sleep(0.4)  # Delay for streaming effect
                yield current_response
            
            # Get actual response
            if use_tools:
                # Use Task Agent for tool-based queries
                print(f"🛠️ Using Task Agent for: {message}")
                
                # Show tool execution in real-time
                current_response += "\n🔧 **Executing Tools...**\n"
                yield current_response
                
                result = self.task_agent.run(message)
                response = result.get("result", "Task completed.")
                
                # Show tool results in real-time
                current_response += f"\n✅ **Tool Results:**\n{response}\n"
                yield current_response
                
                # Add tool execution info
                if "TOOL:" in response:
                    response = f"🛠️ **Tool Execution Complete:**\n\n{response}"
                
            elif any(keyword in message.lower() for keyword in ['knowledge', 'learn', 'about', 'what is', 'explain']):
                # Use RAG Agent for knowledge queries
                print(f"🔍 Using RAG Agent for: {message}")
                result = self.rag_agent.run(message)
                response = result.get("answer", "No relevant information found.")
                
                # Add sources if available
                sources = result.get("sources", [])
                if sources:
                    if isinstance(sources, list):
                        sources_text = "\n".join([f"- {source}" for source in sources[:3]])
                    else:
                        sources_text = f"- {sources}"
                    response += f"\n\n📚 **Sources:**\n{sources_text}"
                
            else:
                # Use general model for regular chat
                print(f"💬 Using General Assistant for: {message}")
                # Format message as list of message dictionaries
                messages = [{"role": "user", "content": message}]
                model_response = self.model_loader.generate_response(messages)
                
                # Handle different response formats
                if isinstance(model_response, dict):
                    response = model_response.get("content", str(model_response))
                elif isinstance(model_response, str):
                    response = model_response
                else:
                    response = str(model_response)
                
                # Format capabilities response better
                if "tools" in message.lower() or "capabilities" in message.lower() or "help" in message.lower():
                    response = self._format_capabilities_response(response)
            
            # Add response header and stream the final response
            current_response += "\n**💡 Response:**\n\n"
            yield current_response
            
            # Stream response word by word with better formatting
            words = response.split()
            for i in range(len(words)):
                current_response += words[i] + " "
                time.sleep(0.08)  # Delay for word-by-word streaming
                yield current_response
            
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            error_response = f"❌ Sorry, I encountered an error: {e}"
            yield error_response

def main():
    """Launch the enhanced Hero Project chat interface with sidebar"""
    print("🚀 Starting Hero Project Enhanced Chat Interface...")
    
    # Create interface instance
    hero_interface = SimpleHeroInterface()
    
    # Custom CSS for better styling
    css = """
    .sidebar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .model-info {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .example-btn {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .example-btn:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-2px);
    }
    .thinking-animation {
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .processing-indicator {
        background: linear-gradient(90deg, #4CAF50, #8BC34A, #4CAF50);
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    """
    
    with gr.Blocks(css=css, title="🤖 Hero Project AI") as demo:
        gr.Markdown("# 🤖 Hero Project AI - Enhanced Interface")
        gr.Markdown("Local AI Assistant with Atomic Agents, RAG, and Tool Calling")
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat with Hero Project AI",
                    type="messages",
                    height=600
                )
                
                msg_input = gr.Textbox(
                    placeholder="Enter your message and press enter...",
                    label="Message",
                    interactive=True
                )
                
                with gr.Column():
                    send_btn = gr.Button("📤 Send", variant="primary", interactive=True)
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", visible=False)
                
                # Clear button
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            
            # Sidebar with model info and examples
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 📊 Model Information", elem_classes="sidebar")
                    
                    with gr.Group(elem_classes="model-info"):
                        gr.Markdown("""
                        **🤖 Model:** Qwen3-4B-Instruct  
                        **📦 Format:** GGUF (4-bit)  
                        **🧠 Context:** 4K tokens  
                        **⚡ Framework:** Atomic Agents  
                        **🗄️ Database:** ChromaDB  
                        **💻 Processing:** Local CPU  
                        **🔒 Privacy:** 100% Local  
                        """)
                
                with gr.Group():
                    gr.Markdown("## 🚀 Try These Examples", elem_classes="sidebar")
                    
                    example_buttons = [
                        gr.Button("👋 Hello! What can you help me with?", elem_classes="example-btn"),
                        gr.Button("📖 Read the README.md file", elem_classes="example-btn"),
                        gr.Button("🔍 Search for information about AI", elem_classes="example-btn"),
                        gr.Button("✍️ Write a test file called hello.txt", elem_classes="example-btn"),
                        gr.Button("🧠 What do you know about machine learning?", elem_classes="example-btn"),
                        gr.Button("🛠️ What tools do you have available?", elem_classes="example-btn")
                    ]
                
        
        # Event handlers
        def chat_with_agent_wrapper(message, history):
            # Add user message to history
            history.append({"role": "user", "content": message})
            
            # Disable input, disable send button, and show stop button
            yield history, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True)
            
            # Get response from agent with real-time updates
            full_response = ""
            for response in hero_interface.chat_with_agent(message, history):
                full_response = response
                # Update history with current response
                current_history = history + [{"role": "assistant", "content": full_response}]
                yield current_history, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True)
            
            # Final update with complete response
            history.append({"role": "assistant", "content": full_response})
            # Re-enable input, re-enable send button, and hide stop button
            yield history, gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=False)
        
        def clear_chat():
            return [], gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=False)
        
        def stop_generation():
            # Re-enable input, re-enable send button, and hide stop button
            return gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=False)
        
        # Connect events
        msg_input.submit(
            chat_with_agent_wrapper,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, send_btn, stop_btn]
        ).then(
            lambda: "",  # Clear input
            outputs=[msg_input]
        )
        
        send_btn.click(
            chat_with_agent_wrapper,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, send_btn, stop_btn]
        ).then(
            lambda: "",  # Clear input
            outputs=[msg_input]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, msg_input, send_btn, stop_btn]
        )
        
        stop_btn.click(
            stop_generation,
            outputs=[msg_input, send_btn, stop_btn]
        )
        
        # Example button handlers
        for i, btn in enumerate(example_buttons):
            example_texts = [
                "Hello! What can you help me with?",
                "Read the README.md file",
                "Search for information about AI",
                "Write a test file called hello.txt",
                "What do you know about machine learning?",
                "What tools do you have available?"
            ]
            btn.click(
                lambda text=example_texts[i]: text,
                outputs=[msg_input]
            )
    
    # Launch the interface
    print("🌐 Launching enhanced web interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7868,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
