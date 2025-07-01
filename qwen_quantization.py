from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
import torch

bnb_config_4 = BitsAndBytesConfig(
    load_in_4bit=True,               # or load_in_8bit=True for 8-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # torch.float16 also works
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",       # or "fp4"
)

bnb_config_8 = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Optional, default value
    llm_int8_skip_modules=None,  # Optional: specify layers to skip
    llm_int8_enable_fp32_cpu_offload=False  # Optional: only for low-memory CPUs
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    quantization_config=bnb_config_8,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
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

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
