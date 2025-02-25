import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# - Set args
model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf"

# - Load model
print("INFO: Loading model %s ..." % (model_id))
model= LlavaOnevisionForConditionalGeneration.from_pretrained(
	model_id, 
	torch_dtype=torch.float16, 
	low_cpu_mem_usage=True, 
	device_map="auto"
)

print("model.parameters")
print(model.parameters)
print("")

# - Print layer names
print("== LAYER NAMES ==")
for item in model.named_parameters():
	print(item[0])
