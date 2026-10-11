from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAMES = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-8B"]
output_dict = {}

for name in MODEL_NAMES:
    path = f"../weights/{name}"
    snapshot_download(repo_id=f"Qwen/{name}", local_dir=path)


    tokenize = AutoTokenizer.from_pretrained(path)
    inputs = tokenize("The capital of France is", return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)

    model = model.to("mps")
    inputs = inputs.to("mps")

    residue = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    output = tokenize.decode(residue[0])
    output_dict[name] = output

    print(f"{name}: {output}")

for key, val in output_dict.items():
    print(f"{key}: {val}")
