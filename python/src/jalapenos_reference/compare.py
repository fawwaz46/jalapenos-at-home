import torch

def validate_inputs(actual:torch.Tensor, expected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual: {actual.shape} ||| expected: {expected.shape}")
    if not actual.any():
        raise ValueError("actual tensor is zeroed out") 
    if not expected.any():
        raise ValueError("expected tensor is zeroed out")
    
    return (actual.double(), expected.double())

def max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    tensors = validate_inputs(actual, expected)
    abs_diff = (tensors[0] - tensors[1]).abs().max().item()

    return abs_diff

def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = validate_inputs(actual, expected)
    error = ((actual-expected).norm()/expected.norm()).item()
    return error

def cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = validate_inputs(actual, expected)
    cos = ((actual * expected).sum() / (actual.norm() * expected.norm())).item()
    return cos

def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    max_diff = max_abs_diff(actual, expected)
    rel_err = relative_error(actual, expected)
    cos = cosine_similarity(actual, expected)

    print(f"maximum absolute difference: {max_diff}")
    print(f"relative error = {rel_err}")
    print(f"cosine similarity: {cos}")

    res = {"max_abs": max_diff,
           "relative_err": rel_err,
           "cos_similarity": cos}

    return res
