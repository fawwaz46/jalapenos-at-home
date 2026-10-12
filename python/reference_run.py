from jalapenos_reference.compare import compare
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

PROMPT="To make a simple loaf of bread at home, you will need flour, water, salt, and"
PATH = "../weights/Qwen3-1.7B"

tokenize = AutoTokenizer.from_pretrained(PATH)
inputs = tokenize(PROMPT, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16)
model = model.to("mps")
inputs = inputs.to("mps")

for b in range(2):
    raw_output = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    output = tokenize.decode(raw_output[0])

    res = model(input_ids=raw_output, output_hidden_states=True)

    tot_states = len(res.hidden_states)
    shape = res.hidden_states[0].shape

    print(tot_states)
    print(shape)
    print(output)

    reference = {}

    reference["token_ids"] = raw_output.cpu().contiguous()
    reference["logits"] = res.logits.cpu().contiguous()

    for n,tensor in enumerate(res.hidden_states):
        reference[f"hidden_{n}"] = tensor.cpu().contiguous()

    save_file(reference, f"../data/reference_qwen3-1.7b_{b}.safetensors")

run0 = load_file("../data/reference_qwen3-1.7b_0.safetensors")
run1 = load_file("../data/reference_qwen3-1.7b_1.safetensors")

for key, val in run0.items():
    result = compare(val, run1[key])

    print(f"{key}: {result}")
