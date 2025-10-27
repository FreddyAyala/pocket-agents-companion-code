#!/usr/bin/env python3
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def chat_with_tinyllama(message, max_length=200):
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

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "Real TinyLlama API",
        "status": "online",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "endpoints": ["/chat - POST: Send messages to the AI", "/health - GET: Check API health"]
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400

    response = chat_with_tinyllama(data['message'])
    return jsonify({"message": data['message'], "response": response, "model": "TinyLlama"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
