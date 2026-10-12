from jalapenos_reference.compare import compare
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM

PATH = "../weights/Qwen3-1.7B"
DRAFT_PATH = "../weights/Qwen3-0.6B"

tok_dict = load_file("../data/reference_qwen3-1.7b_0.safetensors")
tokens = tok_dict["token_ids"][ : , :20]

draft = AutoModelForCausalLM.from_pretrained(DRAFT_PATH, dtype=torch.bfloat16)
target = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16)

draft.to("mps")
target.to("mps")
tokens = tokens.to("mps")

accepted, drafted, rounds = 0, 0, 0

while tokens.shape[1] < 35:
    draft_tokens = draft.generate(tokens, max_new_tokens=5, do_sample=False) #generates token guesses from draft model
    target_check = target(draft_tokens) #generates what the target model think should come next at every token, on what the draft model guessed + prompt
    target_picks = target_check.logits.argmax(dim=-1) #generates the token ids oh what the target guessed

    draft_comp = draft_tokens[ : , -5:]
    target_comp = target_picks[ : , -6:]
    valid_tokens = 0

    for n in range(5):
        if target_comp[0, n] == draft_comp[0, n]:
            valid_tokens += 1
        else:
            break

    tokens = torch.cat([tokens, draft_comp[ : , :valid_tokens], target_comp[ : , valid_tokens:valid_tokens + 1]], dim=1)

    accepted += valid_tokens
    drafted += 5
    rounds += 1

tokens = tokens[ : , :35] # last round can overshoot

match = torch.equal(tokens.cpu(), tok_dict["token_ids"])
print(f"exact match: {match}")
print(f"acceptance rate: {accepted / drafted:.2f}")
print(f"tokens per round: {15 / rounds:.2f} ({rounds} rounds)")
