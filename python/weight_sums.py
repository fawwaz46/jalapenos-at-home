import json
from pathlib import Path

import torch
from safetensors import safe_open

# Answer key for the Rust weight loader: shape + f64 sum of every tensor.
MODEL = "Qwen3-1.7B"
PATH = Path(f"../weights/{MODEL}")
OUT = Path(f"../data/weight_sums_{MODEL.lower()}.json")

sums = {}
for shard in sorted(PATH.glob("*.safetensors")):
    with safe_open(shard, framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            sums[name] = {
                "shape": list(tensor.shape),
                "sum": tensor.to(torch.float64).sum().item(),
            }
    print(f"{shard.name}: done")

OUT.write_text(json.dumps(sums, indent=1, sort_keys=True))
print(f"{len(sums)} tensors -> {OUT}")
