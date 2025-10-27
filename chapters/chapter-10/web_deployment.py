#!/usr/bin/env python3
"""
Chapter 10: Web Deployment with Gradio and Streamlit

This script demonstrates web-native AI deployment using modern web frameworks.
Shows how to create interactive web interfaces for on-device AI.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import time
import threading
import subprocess
import webbrowser
from pathlib import Path

# Web framework imports
try:
    import gradio as gr
    import streamlit as st
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    WEB_FRAMEWORKS_AVAILABLE = True
except ImportError as e:
    WEB_FRAMEWORKS_AVAILABLE = False
    print(f"⚠️ Web frameworks not available: {e}")
    print("Install with: pip install gradio streamlit torch transformers")

class WebDeployment:
    """Web deployment using modern web frameworks"""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize web deployment with a small model"""
        
        if not WEB_FRAMEWORKS_AVAILABLE:
            raise ImportError("Web frameworks are required for web deployment")
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        print(f"🌐 Initializing web deployment with {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the model for web deployment"""
        try:
            print("📥 Loading model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.loaded = True
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.loaded = False
    
    def generate_response(self, message, max_length=200):
        """Generate response for web interface"""
        
        if not self.loaded:
            return "Model not loaded. Please check the setup."
        
        try:
            prompt = f"<|user|>\n{message}\n<|assistant|>\n"
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[-1].strip()
            else:
                response = full_response
            
            response = response.replace("<|user|>", "").replace("<|assistant|>", "").strip()
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def create_gradio_app():
    """Create Gradio web application"""
    
    print("🎨 Creating Gradio application...")
    
    # Initialize deployment
    deployment = WebDeployment()
    
    def gradio_chat(message, history):
        """Chat function for Gradio"""
        response = deployment.generate_response(message)
        return response
    
    # Create Gradio interface
    with gr.Blocks(title="On-Device AI Chat", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 On-Device AI Chat")
        gr.Markdown("Chat with a locally running AI model. No cloud required!")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=True
                )
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    lines=2
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Model Info")
                gr.Markdown(f"**Model:** {deployment.model_name}")
                gr.Markdown(f"**Status:** {'✅ Loaded' if deployment.loaded else '❌ Not Loaded'}")
                gr.Markdown("**Framework:** Gradio")
                gr.Markdown("**Deployment:** Web")
        
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")
        
        # Event handlers
        def user(user_message, history):
            return "", history + [[user_message, None]]
        
        def bot(history):
            user_message = history[-1][0]
            bot_message = deployment.generate_response(user_message)
            history[-1][1] = bot_message
            return history
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        send_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    return app

def create_streamlit_app():
    """Create Streamlit web application"""
    
    print("🎨 Creating Streamlit application...")
    
    # Initialize deployment
    deployment = WebDeployment()
    
    # Streamlit app
    st.set_page_config(
        page_title="On-Device AI Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 On-Device AI Chat")
    st.markdown("Chat with a locally running AI model. No cloud required!")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model:** {deployment.model_name}")
        st.write(f"**Status:** {'✅ Loaded' if deployment.loaded else '❌ Not Loaded'}")
        st.write("**Framework:** Streamlit")
        st.write("**Deployment:** Web")
        
        if st.button("🔄 Refresh Model"):
            st.rerun()
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = deployment.generate_response(prompt)
            st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

def launch_web_applications():
    """Launch web applications in separate processes"""
    
    print("🚀 Launching web applications...")
    print("=" * 70)
    
    # Create applications
    gradio_app = create_gradio_app()
    
    # Launch Gradio app
    print("🌐 Starting Gradio application...")
    print("   URL: http://localhost:7860")
    
    # Launch in a separate thread
    def run_gradio():
        gradio_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    
    gradio_thread = threading.Thread(target=run_gradio, daemon=True)
    gradio_thread.start()
    
    # Wait a moment for Gradio to start
    time.sleep(3)
    
    # Launch Streamlit app
    print("🌐 Starting Streamlit application...")
    print("   URL: http://localhost:8501")
    
    # Create a simple script to run Streamlit
    streamlit_script = """
import streamlit as st
from web_deployment import create_streamlit_app

# This will be handled by the Streamlit app itself
st.set_page_config(page_title="On-Device AI Chat", page_icon="🤖", layout="wide")

# Initialize deployment
from web_deployment import WebDeployment
deployment = WebDeployment()

# Streamlit app code here (same as create_streamlit_app function)
"""
    
    # Save Streamlit script
    with open("streamlit_app.py", "w") as f:
        f.write(streamlit_script)
    
    # Launch Streamlit in a separate process
    streamlit_process = subprocess.Popen([
        "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
    
    print("\n✅ Web applications launched!")
    print("📱 Access your applications:")
    print("   • Gradio: http://localhost:7860")
    print("   • Streamlit: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop all applications")
    
    try:
        # Keep the main process running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping web applications...")
        streamlit_process.terminate()
        print("✅ Applications stopped")

def demonstrate_web_deployment():
    """Demonstrate web deployment capabilities"""
    
    print("=" * 70)
    print("Chapter 10: Web Deployment Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    if not WEB_FRAMEWORKS_AVAILABLE:
        print("❌ Web frameworks not available")
        print("💡 Install with: pip install gradio streamlit torch transformers")
        return
    
    try:
        # Test model loading
        print("🧪 Testing web deployment...")
        deployment = WebDeployment()
        
        if deployment.loaded:
            # Test generation
            test_prompt = "Hello! How are you?"
            print(f"📝 Testing with: {test_prompt}")
            response = deployment.generate_response(test_prompt)
            print(f"🤖 Response: {response}")
            
            print("\n✅ Web deployment ready!")
            print("💡 Key advantages:")
            print("   • Interactive web interfaces")
            print("   • Real-time chat experience")
            print("   • No cloud dependencies")
            print("   • Easy deployment and sharing")
            
            # Ask if user wants to launch web apps
            print("\n🌐 Would you like to launch web applications?")
            print("   This will start Gradio and Streamlit servers")
            
        else:
            print("❌ Model loading failed")
            
    except Exception as e:
        print(f"❌ Error in web deployment demo: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        launch_web_applications()
    else:
        demonstrate_web_deployment()
