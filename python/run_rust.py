import argparse
import json
from pathlib import Path
import subprocess
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--backend", choices=["cpu", "metal", "gpu"], default="metal")
    parser.add_argument("--model", default="Qwen3-1.7B")
    parser.add_argument("--tokens", type=int, default=15)
    parser.add_argument("--draft", nargs="?", const="Qwen3-0.6B")
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--mxfp4", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--max-gamma", type=int, default=8)
    parser.add_argument("--prefill-backend", choices=["cpu", "metal", "gpu"])
    parser.add_argument("--draft-backend", choices=["cpu", "metal", "gpu"])
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "weights" / args.model)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    command = [ROOT / "engine/target/release/engine", args.backend, args.model, str(args.tokens)]
    if args.mxfp4:
        command += ["--mxfp4"]
    if args.draft:
        command += ["--draft", args.draft, "--gamma", str(args.gamma)]
    if args.adaptive:
        command += ["--adaptive", "--max-gamma", str(args.max_gamma)]
    if args.prefill_backend:
        command += ["--prefill-backend", args.prefill_backend]
    if args.draft_backend:
        command += ["--draft-backend", args.draft_backend]
    if args.stats:
        command += ["--stats"]
    result = subprocess.run(
        command,
        input=json.dumps(prompt),
        text=True,
        capture_output=True,
    )
    if result.returncode:
        parser.exit(result.returncode, result.stderr)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    print(tokenizer.decode(json.loads(result.stdout), skip_special_tokens=True))


if __name__ == "__main__":
    main()
