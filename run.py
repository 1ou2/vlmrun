#!/usr/bin/env python3
"""
Test script for Qwen2.5-VL-32B-Instruct model
Requires: transformers, torch, pillow, requests
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO
import time

def setup_model():
    """Load the Qwen2.5-VL model and processor"""
    print("Loading Qwen2.5-VL-32B-Instruct model...")
    print("This may take a few minutes for the first time...")
    
    model_name = "Qwen/Qwen2.5-VL-32B-Instruct"
    
    # Load model with optimizations for your L40S
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on L40S
        attn_implementation="flash_attention_2",  # Enable flash attention
        device_map="cuda:0",  # Single GPU
        trust_remote_code=True
    )
    
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, processor

def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def load_image_from_path(path):
    """Load image from local path"""
    try:
        image = Image.open(path)
        return image
    except Exception as e:
        print(f"Error loading image from path: {e}")
        return None

def run_inference(model, processor, image, text_prompt):
    """Run inference on image with text prompt"""
    
    # Prepare messages in the expected format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    
    # Prepare inputs for the model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    print("Running inference...")
    start_time = time.time()
    
    # Generate response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    return response, inference_time

def main():
    print("=" * 60)
    print("Qwen2.5-VL-32B-Instruct Test Script")
    print("=" * 60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Load model
    try:
        model, processor = setup_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test images - you can modify these
    test_cases = [
        {
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1200px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            "prompt": "Describe this image in detail."
        },
        {
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
            "prompt": "What do you see in this image? What colors and shapes are present?"
        }
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} Test {i} {'='*20}")
        print(f"Image URL: {test_case['image_url']}")
        print(f"Prompt: {test_case['prompt']}")
        print("-" * 60)
        
        # Load image
        image = load_image_from_url(test_case['image_url'])
        if image is None:
            print("Failed to load image, skipping...")
            continue
            
        print(f"Image loaded: {image.size} pixels, mode: {image.mode}")
        
        # Run inference
        try:
            response, inference_time = run_inference(
                model, processor, image, test_case['prompt']
            )
            
            print(f"\n🤖 Model Response:")
            print(f"{response}")
            print(f"\n⏱️  Inference time: {inference_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
    
    # Memory usage info
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.max_memory_reserved(0) / 1024**3
        print(f"\n📊 GPU Memory Usage:")
        print(f"   Peak allocated: {memory_used:.2f} GB")
        print(f"   Peak cached: {memory_cached:.2f} GB")

if __name__ == "__main__":
    main()