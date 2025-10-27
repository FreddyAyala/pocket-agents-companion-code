#!/usr/bin/env python3
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
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

def gradio_chat(message, history):
    response = chat_with_tinyllama(message)
    return response

# Create interface
with gr.Blocks(title="Real TinyLlama AI Assistant") as demo:
    gr.Markdown("# 🤖 Real TinyLlama On-Device AI Assistant")
    gr.Markdown("**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 | **Parameters**: 1.1B | **Provider**: TinyLlama")

    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot(label="Real AI Conversation", height=400, type="messages")
        msg = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [{"role": "user", "content": user_message}]

        def bot(history):
            user_message = history[-1]["content"]
            bot_message = chat_with_tinyllama(user_message)
            history.append({"role": "assistant", "content": bot_message})
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
