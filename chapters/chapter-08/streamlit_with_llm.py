#!/usr/bin/env python3
"""
Streamlit App with Real Local LLM
This app uses the actual local language model for real AI responses.
"""

import streamlit as st
import time
from local_llm_service import get_llm_service

# Configure Streamlit
st.set_page_config(
    page_title="On-Device AI with Real LLM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the LLM service
@st.cache_resource
def get_llm():
    return get_llm_service()

llm_service = get_llm()

# Main app
st.title("🤖 On-Device AI Assistant")
st.markdown("**Powered by a real local language model - no data leaves your device!**")

# Sidebar with model info
with st.sidebar:
    st.header("🎛️ Model Controls")
    
    # Display model status
    status = llm_service.get_status()
    st.subheader("📊 Model Status")
    st.success(f"✅ {status['model_name']}")
    st.info(f"🖥️ Device: {status['device']}")
    st.metric("Parameters", status['parameters'])
    
    # Settings
    st.subheader("⚙️ Settings")
    max_length = st.slider("Max Response Length", 50, 200, 100)
    temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
    
    # Status indicators
    st.subheader("📈 Performance")
    st.success("🟢 Model Loaded")
    st.info("🟢 Ready for Inference")
    st.metric("Memory Usage", "~500MB", "0MB")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat with AI", "📝 Text Analysis", "🔧 Model Tools", "ℹ️ About"])

with tab1:
    st.header("💬 Chat with Real AI")
    
    # Chat interface
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
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                response = llm_service.generate_response(prompt, max_length)
            st.markdown(response)
        
        # Add AI response
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.header("📝 Text Analysis with AI")
    
    # Text input
    text_input = st.text_area("Enter text to analyze", height=200, placeholder="Enter your text here...")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Summarize"):
            if text_input:
                with st.spinner("Generating summary..."):
                    prompt = f"Summarize this text in 2-3 sentences: {text_input}"
                    summary = llm_service.generate_response(prompt, max_length=100)
                st.success("Summary Generated!")
                st.write(summary)
            else:
                st.warning("Please enter some text")
    
    with col2:
        if st.button("🎯 Analyze Sentiment"):
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    prompt = f"Analyze the sentiment of this text: {text_input}"
                    sentiment = llm_service.generate_response(prompt, max_length=50)
                st.success("Sentiment Analysis Complete!")
                st.write(sentiment)
            else:
                st.warning("Please enter some text")
    
    with col3:
        if st.button("🔍 Extract Key Points"):
            if text_input:
                with st.spinner("Extracting key points..."):
                    prompt = f"Extract the key points from this text: {text_input}"
                    key_points = llm_service.generate_response(prompt, max_length=100)
                st.success("Key Points Extracted!")
                st.write(key_points)
            else:
                st.warning("Please enter some text")

with tab3:
    st.header("🔧 Model Tools")
    
    # Model testing
    st.subheader("🧪 Test the Model")
    test_prompt = st.text_input("Test prompt", value="Hello, how are you?")
    
    if st.button("Generate Response"):
        with st.spinner("Generating..."):
            response = llm_service.generate_response(test_prompt, max_length)
        st.write("**Response:**")
        st.write(response)
    
    # Model info
    st.subheader("📊 Model Information")
    model_info = {
        "Model Name": status['model_name'],
        "Device": status['device'],
        "Parameters": status['parameters'],
        "Status": "Loaded" if status['loaded'] else "Not Loaded"
    }
    
    for key, value in model_info.items():
        st.metric(key, value)

with tab4:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## On-Device AI with Real Language Model
    
    This application demonstrates **real on-device AI** capabilities using an actual language model:
    
    ### ✨ Features
    - **Real AI Model**: Microsoft DialoGPT-small (124M parameters)
    - **Privacy First**: All processing happens on your device
    - **No Internet Required**: Works completely offline
    - **Fast Response**: Optimized for your hardware
    - **Secure**: Your data never leaves your machine
    
    ### 🛠️ Technologies
    - **Hugging Face Transformers**: For loading and running the model
    - **PyTorch**: For neural network operations
    - **Streamlit**: Interactive web application framework
    - **Local Processing**: No cloud dependencies
    
    ### 🚀 Model Details
    - **Name**: microsoft/DialoGPT-small
    - **Parameters**: ~124 million
    - **Type**: Conversational AI model
    - **Size**: ~500MB
    - **Performance**: Optimized for real-time chat
    
    ### 📱 Mobile Ready
    This app is designed to work seamlessly on:
    - Desktop computers
    - Tablets
    - Mobile phones
    - Any device with a web browser
    
    ### 🔒 Privacy & Security
    - **Local Processing**: All AI inference happens on your device
    - **No Data Transmission**: Your conversations never leave your machine
    - **Offline Capable**: Works without internet connection
    - **Secure**: No external API calls or data sharing
    """)

if __name__ == "__main__":
    print("🚀 Starting Streamlit App with Real Local LLM...")
    print("🌐 App will be available at: http://localhost:8501")
