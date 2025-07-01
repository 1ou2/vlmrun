# ultra_simple_test.py
from transformers import pipeline
import torch

print("Testing with pipeline...")

# Create a vision-language pipeline
pipe = pipeline(
    "image-to-text",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

# Test with a URL
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

result = pipe(image_url, prompt="Describe this image.")
print("Result:", result)