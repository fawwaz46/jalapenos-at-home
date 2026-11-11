"""Record drafting runs for the site's visualizer.

The engine only logs how many guesses were accepted, so this replays the
0.6B drafter in Transformers to get the rejected tokens too.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "src" / "data" / "runs.json"
PROMPTS = [
    "To make a simple loaf of bread",
    "The fastest way to learn a new language is",
    "A GPU is different from a CPU because",
]
TOKENS = 48


def run_engine(prompt_ids):
    command = [
        ROOT / "engine/target/release/engine", "gpu", "Qwen3-1.7B", str(TOKENS),
        "--draft", "Qwen3-0.6B", "--gamma", "5", "--adaptive", "--max-gamma", "8", "--stats",
    ]
    result = subprocess.run(command, input=json.dumps(prompt_ids), text=True, capture_output=True)
    if result.returncode:
        sys.exit(result.stderr)
    stats = next(json.loads(line) for line in result.stderr.splitlines() if line.startswith("{"))
    return json.loads(result.stdout), stats


@torch.no_grad()
def draft_greedy(model, prefix, count):
    ids = torch.tensor([prefix], device=model.device)
    out = []
    for _ in range(count):
        token = int(model(ids).logits[0, -1].argmax())
        out.append(token)
        ids = torch.cat([ids, torch.tensor([[token]], device=model.device)], dim=1)
    return out


def main():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "weights" / "Qwen3-1.7B")
    drafter = AutoModelForCausalLM.from_pretrained(
        ROOT / "weights" / "Qwen3-0.6B", dtype=torch.float32
    ).to("mps")
    piece = lambda t: tokenizer.decode([t])
    runs = []
    for text in PROMPTS:
        prompt = tokenizer.encode(text, add_special_tokens=False)
        generated, stats = run_engine(prompt)
        rounds, p, mismatches = [], 0, 0
        for r in stats["round_stats"]:
            accepted, drafted = r["accepted"], r["drafted"]
            proposed = draft_greedy(drafter, prompt + generated[:p], drafted)
            ok = proposed[:accepted] == generated[p : p + accepted] and (
                accepted == drafted
                or p + accepted >= len(generated)
                or proposed[accepted] != generated[p + accepted]
            )
            mismatches += not ok
            target = generated[p + accepted] if p + accepted < len(generated) else None
            rounds.append({
                "gamma": r["gamma"],
                "next_gamma": r["next_gamma"],
                "accepted": accepted,
                "proposed": [piece(t) for t in proposed],
                "target": None if target is None else piece(target),
                "bonus": accepted == drafted,
            })
            p += accepted + (target is not None)
        runs.append({
            "prompt": text,
            "output": tokenizer.decode(generated),
            "accepted": stats["accepted"],
            "drafted": stats["drafted"],
            "verify_passes": stats["verify"]["passes"],
            "draft_passes": stats["draft"]["passes"],
            "elapsed_ms": stats["elapsed_ms"],
            "rounds": rounds,
            "replay_mismatches": mismatches,
        })
        print(f"{text!r}: {stats['accepted']}/{stats['drafted']} accepted, "
              f"{len(rounds)} rounds, replay mismatches {mismatches}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": "Qwen3-1.7B", "draft": "Qwen3-0.6B", "backend": "gpu",
                               "precision": "bf16", "tokens": TOKENS, "runs": runs}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
