# simple_test.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests

print("Loading model...")
model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

# Simple loading approach
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
).eval()

print("Model loaded successfully!")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")