#!/usr/bin/env python3
"""
Comprehensive quantization options for Qwen2.5-VL models
Shows different quantization methods with memory usage estimates
"""

import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO
import os

def test_quantization_method(model_name, method_name, quantization_config=None, torch_dtype=None, device_map="auto"):
    """Test a specific quantization method"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {method_name}")
    print(f"{'='*60}")
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        
        if quantization_config:
            kwargs["quantization_config"] = quantization_config
        if torch_dtype:
            kwargs["torch_dtype"] = torch_dtype
            
        # Load model
        print("📥 Loading model...")
        model = AutoModel.from_pretrained(model_name, **kwargs)
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Check memory usage
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        
        print(f"✅ {method_name} loaded successfully!")
        print(f"📊 GPU Memory used: {memory_used:.2f} GB")
        print(f"📊 GPU Memory free: {memory_free:.2f} GB")
        
        # Quick inference test
        print("🔄 Quick inference test...")
        
        # Use a small test image
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content))
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What do you see?"}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        print(f"🤖 Response: {response[:100]}...")
        
        # Final memory check
        peak_memory = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"📊 Peak GPU Memory: {peak_memory:.2f} GB")
        
        # Cleanup
        del model, processor, inputs
        torch.cuda.empty_cache()
        
        return True, memory_used, peak_memory
        
    except Exception as e:
        print(f"❌ {method_name} failed: {str(e)[:100]}...")
        torch.cuda.empty_cache()
        return False, 0, 0

def main():
    print("🚀 Comprehensive Quantization Testing for Qwen2.5-VL")
    print("=" * 80)
    
    # Set memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test with different model sizes
    models_to_test = [
        "Qwen/Qwen2.5-VL-7B-Instruct",    # Start with smaller model
        "Qwen/Qwen2.5-VL-14B-Instruct",   # Medium model
        # "Qwen/Qwen2.5-VL-32B-Instruct", # Uncomment if you want to test 32B
    ]
    
    for model_name in models_to_test:
        print(f"\n🎯 Testing model: {model_name}")
        print("=" * 80)
        
        results = []
        
        # 1. No Quantization (Baseline)
        success, mem_used, peak_mem = test_quantization_method(
            model_name, 
            "No Quantization (bfloat16)", 
            torch_dtype=torch.bfloat16
        )
        if success:
            results.append(("No Quantization", mem_used, peak_mem))
        
        # 2. 8-bit Quantization (LLM.int8())
        quantization_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        success, mem_used, peak_mem = test_quantization_method(
            model_name, 
            "8-bit Quantization (LLM.int8())", 
            quantization_config=quantization_8bit
        )
        if success:
            results.append(("8-bit LLM.int8()", mem_used, peak_mem))
        
        # 3. 4-bit NF4 Quantization (QLoRA)
        quantization_4bit_nf4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4
        )
        success, mem_used, peak_mem = test_quantization_method(
            model_name, 
            "4-bit NF4 Quantization (QLoRA)", 
            quantization_config=quantization_4bit_nf4
        )
        if success:
            results.append(("4-bit NF4", mem_used, peak_mem))
        
        # 4. 4-bit FP4 Quantization
        quantization_4bit_fp4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",  # Float4
        )
        success, mem_used, peak_mem = test_quantization_method(
            model_name, 
            "4-bit FP4 Quantization", 
            quantization_config=quantization_4bit_fp4
        )
        if success:
            results.append(("4-bit FP4", mem_used, peak_mem))
        
        # 5. 4-bit with different compute dtype
        quantization_4bit_fp16 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Use fp16 instead of bfloat16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        success, mem_used, peak_mem = test_quantization_method(
            model_name, 
            "4-bit NF4 + FP16 compute", 
            quantization_config=quantization_4bit_fp16
        )
        if success:
            results.append(("4-bit NF4 + FP16", mem_used, peak_mem))
        
        # 6. 4-bit without double quantization
        quantization_4bit_single = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,  # Single quantization
            bnb_4bit_quant_type="nf4",
        )
        success, mem_used, peak_mem = test_quantization_method(
            model_name, 
            "4-bit NF4 (Single Quantization)", 
            quantization_config=quantization_4bit_single
        )
        if success:
            results.append(("4-bit NF4 Single", mem_used, peak_mem))
        
        # Print results summary
        if results:
            print(f"\n📊 MEMORY USAGE SUMMARY for {model_name}")
            print("=" * 80)
            print(f"{'Method':<25} {'Used (GB)':<12} {'Peak (GB)':<12} {'Savings':<10}")
            print("-" * 80)
            
            baseline_mem = results[0][1] if results else 0
            for method, used, peak in results:
                savings = f"{((baseline_mem - used) / baseline_mem * 100):.1f}%" if baseline_mem > 0 else "N/A"
                print(f"{method:<25} {used:<12.2f} {peak:<12.2f} {savings:<10}")
        
        print(f"\n✅ Completed testing {model_name}")
    
    print("\n" + "="*80)
    print("🎉 QUANTIZATION TESTING COMPLETE!")
    print("="*80)
    print("\n📝 RECOMMENDATIONS:")
    print("• 4-bit NF4 with double quantization: Best memory savings with good quality")
    print("• 8-bit LLM.int8(): Good balance of memory and quality")
    print("• 4-bit FP4: Alternative to NF4, sometimes better for certain models")
    print("• No quantization: Best quality but highest memory usage")
    
    print(f"\n💡 TIPS:")
    print("• NF4 (NormalFloat4) generally performs better than FP4 for language models")
    print("• Double quantization saves additional ~0.4 bits per parameter")
    print("• bfloat16 compute dtype usually works better than float16")
    print("• 8-bit quantization preserves quality better but saves less memory than 4-bit")

if __name__ == "__main__":
    main()