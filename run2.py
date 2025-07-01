#!/usr/bin/env python3
"""
Simple test for Qwen2.5-VL-32B-Instruct model
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO

def main():
    print("🚀 Testing Qwen2.5-VL-32B-Instruct")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    #model_name = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    print(f"\n📥 Loading model: {model_name}")
    print("This will take a few minutes...")
    
    try:
        # Use AutoModel instead of specific class to avoid compatibility issues
        """
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True
        )
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True
        )
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True
        )

        # Load processor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        print("✅ Model loaded successfully!")
        
        # Check memory usage
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"📊 GPU Memory used: {memory_used:.2f} GB")
        
        # Test with a simple image
        print("\n🖼️  Loading test image...")
        
        # Use a simple test image
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        
        try:
            response = requests.get(image_url, timeout=10)
            image_path = "test_image.jpg"
            image = Image.open(image_path)
            print(f"✅ Local image loaded: {image.size}")
            
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
            inputs = inputs.to("cuda")
            
            print("🧠 Generating response...")
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id
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
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()