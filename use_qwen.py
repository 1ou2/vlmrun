from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
QWEN="Qwen/Qwen2.5-VL-32B-Instruct"
MAX_NEW_TOKENS=128
FILE = "file:///home/ubuntu/vlmrun/tmp/page_01.jpg"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN, 
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa", #"flash_attention_2"
    device_map="auto"
)


# default processer
processor = AutoProcessor.from_pretrained(QWEN,use_fast=True)
processor.save_pretrained("./my_qwen_processor")
processor = AutoProcessor.from_pretrained("./my_qwen_processor")


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
                "image": FILE,
            },
            {"type": "text", "text": "Décris en détail cette image."},
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
generated_ids = model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
