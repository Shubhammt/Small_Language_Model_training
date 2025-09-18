import re
import reasoning_gym
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print
import torch

model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

loc = 'C:\Users\729sh\.cache\huggingface\hub'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))