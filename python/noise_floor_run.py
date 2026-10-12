from jalapenos_reference.compare import compare
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

PROMPT="To make a simple loaf of bread at home, you will need flour, water, salt, and"
PATH = "../weights/Qwen3-1.7B"
ref1 = load_file("../data/reference_qwen3-1.7b_0.safetensors")

model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32)
model.to("mps")

res = model(ref1["token_ids"].to("mps"), output_hidden_states=True)

reference = {}
reference["logits"] = res.logits.cpu().contiguous()

for n, tensor in enumerate(res.hidden_states):
    reference[f"hidden_{n}"] = tensor.cpu().contiguous()

keys = [f"hidden_{n}" for n in range(29)] + ["logits"]

for key in keys:
    result = compare(ref1[key], reference[key])
    print(f"{key}: {result}")

    