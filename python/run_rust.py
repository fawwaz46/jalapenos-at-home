import argparse
import json
from pathlib import Path
import subprocess

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--backend", choices=["cpu", "metal"], default="metal")
    parser.add_argument("--model", default="Qwen3-1.7B")
    parser.add_argument("--tokens", type=int, default=15)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "weights" / args.model)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    result = subprocess.run(
        [ROOT / "engine/target/release/engine", args.backend, args.model, str(args.tokens)],
        input=json.dumps(prompt),
        text=True,
        capture_output=True,
    )
    if result.returncode:
        parser.exit(result.returncode, result.stderr)
    print(tokenizer.decode(json.loads(result.stdout), skip_special_tokens=True))


if __name__ == "__main__":
    main()
