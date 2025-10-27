#!/usr/bin/env python3
"""
Gradio Interface with Real Local LLM
This interface uses the actual local language model for real AI responses.
"""

import gradio as gr
import time
from local_llm_service import get_llm_service

# Get the LLM service
llm_service = get_llm_service()

def chat_with_llm(message, history):
    """Chat function using real LLM"""
    if not message.strip():
        return "", history
    
    # Generate response using real LLM
    response = llm_service.generate_response(message, max_length=150)
    
    # Update history
    history.append([message, response])
    return "", history

def summarize_with_llm(text):
    """Summarize text using real LLM"""
    if len(text) < 50:
        return "Text is too short to summarize meaningfully."
    
    # Create a summarization prompt
    prompt = f"Please summarize the following text in 2-3 sentences:\n\n{text}"
    summary = llm_service.generate_response(prompt, max_length=100)
    
    return f"Summary: {summary}"

def classify_sentiment(text):
    """Classify sentiment using real LLM"""
    if not text.strip():
        return "Please enter some text to analyze."
    
    # Create a sentiment analysis prompt
    prompt = f"Analyze the sentiment of this text (positive, negative, or neutral): {text}"
    sentiment = llm_service.generate_response(prompt, max_length=50)
    
    return f"Sentiment Analysis: {sentiment}"

# Create Gradio interface
with gr.Blocks(title="On-Device AI with Real LLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 On-Device AI Assistant")
    gr.Markdown("**Powered by a real local language model - no data leaves your device!**")
    
    # Display model status
    status = llm_service.get_status()
    gr.Markdown(f"""
    ### 📊 Model Status
    - **Model**: {status['model_name']}
    - **Device**: {status['device']}
    - **Parameters**: {status['parameters']}
    - **Status**: {'✅ Loaded' if status['loaded'] else '⏳ Loading...'}
    """)
    
    with gr.Tab("💬 Chat with Real AI"):
        chatbot = gr.Chatbot(label="Conversation", height=400)
        msg = gr.Textbox(label="Your message", placeholder="Type your message here...")
        clear = gr.Button("Clear")
        
        msg.submit(chat_with_llm, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    with gr.Tab("📝 Summarize with AI"):
        gr.Markdown("Enter text to get an AI-generated summary using the local model")
        text_input = gr.Textbox(label="Text to summarize", lines=5, placeholder="Enter your text here...")
        summary_output = gr.Textbox(label="AI Summary", lines=3)
        summarize_btn = gr.Button("Summarize with AI")
        
        summarize_btn.click(summarize_with_llm, inputs=text_input, outputs=summary_output)
    
    with gr.Tab("🎯 Sentiment Analysis"):
        gr.Markdown("Analyze the sentiment of your text using the local AI model")
        classify_input = gr.Textbox(label="Text to analyze", lines=3, placeholder="Enter text to analyze...")
        classify_output = gr.Textbox(label="Sentiment Analysis", lines=2)
        classify_btn = gr.Button("Analyze Sentiment")
        
        classify_btn.click(classify_sentiment, inputs=classify_input, outputs=classify_output)
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About This Demo
        
        This interface demonstrates **real on-device AI** capabilities:
        
        - **Real AI Model**: Uses Microsoft DialoGPT-small (124M parameters)
        - **Privacy First**: All processing happens locally on your device
        - **No Internet Required**: Works completely offline
        - **Fast Response**: Optimized for your hardware
        - **Secure**: Your data never leaves your machine
        
        ### Technologies Used:
        - **Hugging Face Transformers**: For loading and running the model
        - **PyTorch**: For neural network operations
        - **Gradio**: Interactive web interface
        - **Local Processing**: No cloud dependencies
        
        ### Model Details:
        - **Name**: microsoft/DialoGPT-small
        - **Parameters**: ~124 million
        - **Type**: Conversational AI model
        - **Size**: ~500MB
        - **Performance**: Optimized for real-time chat
        """)

if __name__ == "__main__":
    print("🚀 Starting Gradio Interface with Real Local LLM...")
    print("🌐 Interface will be available at: http://localhost:7860")
    
    demo.launch(
        server_name='0.0.0.0', 
        server_port=7860, 
        share=False, 
        quiet=False
    )
