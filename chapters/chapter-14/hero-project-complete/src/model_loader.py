"""
Model Loader for Qwen3-4B Vision-Language Model in GGUF format
Hero Project: On-Device AI Agent with Vision and RAG
"""

import torch
import warnings
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from PIL import Image
import base64
import io
import logging
import os

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class Qwen3VLLoader:
    """
    Loader for Qwen3-4B Vision-Language model in GGUF format
    """
    
    def __init__(self, model_path="unsloth/Qwen3-4B-Instruct-2507-GGUF", device=None):
        self.model_path = model_path
        self.model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_file_path = None
        
    def load(self):
        """
        Load the GGUF model
        """
        print(f"🔄 Loading {self.model_path} in GGUF format...")
        
        try:
            # Download the GGUF model file if not already present
            model_filename = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
            
            if not os.path.exists("./models"):
                os.makedirs("./models")
            
            # Check if model already exists locally
            local_model_path = f"./models/{model_filename}"
            if os.path.exists(local_model_path):
                print(f"✅ Using existing model: {local_model_path}")
                self.model_file_path = local_model_path
            else:
                print(f"📥 Downloading model: {self.model_path}")
                self.model_file_path = hf_hub_download(
                    repo_id=self.model_path,
                    filename=model_filename,
                    local_dir="./models",
                    local_dir_use_symlinks=False
                )
                print(f"✅ Model downloaded to: {self.model_file_path}")
            
            # Load the GGUF model
            self.model = Llama(
                model_path=self.model_file_path,
                n_ctx=4096,  # Context window
                n_gpu_layers=-1 if torch.cuda.is_available() else 0,  # Use GPU if available
                verbose=False
            )
            
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"   Model format: GGUF")
            print(f"   Context window: 4096 tokens")
            
            return self.model
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("💡 Make sure you have downloaded the Qwen3-4B GGUF model file")
            raise
    
    def _image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def generate_response(self, messages, images=None, max_new_tokens=512, temperature=0.7):
        """
        Generate response from the GGUF model
        
        Args:
            messages: List of message dictionaries
            images: Optional list of PIL images
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response string
        """
        if self.model is None:
            # Return mock response for demonstration
            user_message = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_message = msg["content"]
                    break
            
            if images and len(images) > 0:
                return f"Mock response: I can see the image you uploaded and understand your question: '{user_message}'. In a real implementation, I would analyze the image and provide a detailed response using the Qwen3-4B model."
            else:
                return f"Mock response: I understand your question: '{user_message}'. In a real implementation, I would provide a detailed response using the Qwen3-4B model. To use the full functionality, please download the GGUF model file."
        
        try:
            prompt_parts = []
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == "user":
                    if images and len(images) > 0:
                        image_b64 = self._image_to_base64(images[0])
                        prompt_parts.append(f"<|im_start|>user\n<|vision_start|><image>{image_b64}</image><|vision_end|>\n{content}<|im_end|>")
                    else:
                        prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == "assistant":
                    prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(prompt_parts)
            
            response = self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_model_info(self):
        """
        Get model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_path": self.model_path,
            "model_file": self.model_file_path,
            "device": self.device,
            "format": "GGUF",
            "context_window": 4096,
            "status": "loaded"
        }