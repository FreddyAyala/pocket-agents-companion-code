#!/usr/bin/env python3
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def chat_with_tinyllama(message, model, tokenizer, max_length=200):
    prompt = f"<|user|>\n{message}\n<|assistant|>\n"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_length=inputs.shape[1] + max_length,
            num_return_sequences=1, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1, no_repeat_ngram_size=3
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[-1].strip()
    else:
        response = full_response
    return response.replace("<|user|>", "").replace("<|assistant|>", "").strip()

st.set_page_config(page_title="Real TinyLlama AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Real TinyLlama On-Device AI Assistant")
st.markdown("**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 | **Parameters**: 1.1B | **Provider**: TinyLlama")

model, tokenizer = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_tinyllama(prompt, model, tokenizer)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
