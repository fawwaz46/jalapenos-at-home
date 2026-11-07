import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize_mxfp4 import dequantize, FORMAT

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "To make a simple loaf of bread at home, you will need flour, water, salt, and"


def rust_tokens(model, prompt, count, backend, draft=None):
    command = [str(ROOT / "engine/target/release/engine"), backend, model, str(count)]
    if draft:
        command += ["--draft", draft]
    output = subprocess.run(command, input=json.dumps(prompt), text=True, capture_output=True, check=True)
    if output.stderr:
        print(output.stderr, end="")
    return json.loads(output.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-1.7B")
    parser.add_argument("--tokens", type=int, default=15)
    parser.add_argument("--backend", choices=["cpu", "metal"], default="cpu")
    parser.add_argument("--draft", default="Qwen3-0.6B")
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "weights" / args.model)
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    inputs = torch.tensor([ids])
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / "weights" / args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.inference_mode():
        baseline = model.generate(inputs, max_new_tokens=args.tokens, do_sample=False)

    # Run Transformers on independently unpacked f32 weights. This checks the
    # whole Rust execution path against a different model implementation.
    for path in sorted((ROOT / "weights" / (args.model + "-MXFP4")).glob("*.safetensors")):
        with safe_open(path, framework="numpy") as weights:
            if weights.metadata()["format"] != FORMAT:
                raise ValueError(f"unexpected checkpoint format: {path}")
            for name in weights.keys():
                if name.endswith(".mxfp4_packed"):
                    parameter_name = name.removesuffix(".mxfp4_packed")
                    values = dequantize(
                        weights.get_tensor(name),
                        weights.get_tensor(parameter_name + ".mxfp4_scales"),
                    )
                    parameter = model.get_parameter(parameter_name)
                    with torch.no_grad():
                        parameter.copy_(torch.from_numpy(values))
    with torch.inference_mode():
        reference = model.generate(inputs, max_new_tokens=args.tokens, do_sample=False)
        hidden = model(inputs, output_hidden_states=True).hidden_states[1][0]
    expected = reference[0, len(ids):].tolist()
    actual = rust_tokens(args.model + "-MXFP4", ids, args.tokens, args.backend)
    assert actual == expected, (actual, expected)
    speculative = rust_tokens(
        args.model + "-MXFP4", ids, args.tokens, args.backend, args.draft + "-MXFP4"
    )
    assert speculative == expected, (speculative, expected)

    fixture = {
        "model": args.model + "-MXFP4",
        "prompt": ids,
        "tokens": expected,
        "hidden_1": hidden.tolist(),
    }
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    (data / f"reference_{args.model.lower()}_mxfp4.json").write_text(json.dumps(fixture))
    print(f"Rust {args.backend} greedy and speculative output match Transformers ({len(expected)} tokens)")
    print(f"BF16 weights in f32 Transformers: {tokenizer.decode(baseline[0, len(ids):])}")
    print(f"MXFP4: {tokenizer.decode(expected)}")


if __name__ == "__main__":
    main()
