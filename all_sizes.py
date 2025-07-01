#!/usr/bin/env python3
"""
Qwen2.5-VL with BitsAndBytes quantization for maximum memory efficiency
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO
import os

def main():
    print("🚀 Testing Qwen2.5-VL with 4-bit Quantization")
    print("=" * 60)
    
    # Set memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Try 32B first, then fall back to smaller models
    models_to_try = [
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-14B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ]
    
    for model_name in models_to_try:
        print(f"\n📥 Loading model: {model_name}")
        print("Using 4-bit quantization for maximum memory efficiency...")
        
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            # Load model with quantization
            model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Load processor
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            print("✅ Model loaded successfully with quantization!")
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            print(f"📊 GPU Memory used: {memory_used:.2f} GB")
            print(f"📊 GPU Memory free: {memory_free:.2f} GB")
            
            # Test inference
            print("\n🖼️  Loading test image...")
            
            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            
            try:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content))
                print(f"✅ Image loaded: {image.size}")
                
                # Prepare the conversation
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image", 
                                "image": image
                            },
                            {
                                "type": "text", 
                                "text": "Describe this image briefly."
                            }
                        ]
                    }
                ]
                
                # Process the input
                print("\n🔄 Processing input...")
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
                
                # Move inputs to GPU
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                print("🧠 Generating response...")
                
                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                
                # Decode response
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                response = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                print("\n" + "="*50)
                print("🤖 MODEL RESPONSE:")
                print("="*50)
                print(response)
                print("="*50)
                
                # Final memory check
                memory_used = torch.cuda.max_memory_allocated(0) / 1024**3
                print(f"\n📊 Peak GPU Memory: {memory_used:.2f} GB")
                print("✅ Test completed successfully!")
                
                return  # Success! Exit
                
            except Exception as e:
                print(f"❌ Error during inference: {e}")
                
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            print("Trying next model...")
            continue
    
    print("❌ All models failed. Check your setup and dependencies.")

if __name__ == "__main__":
    main()