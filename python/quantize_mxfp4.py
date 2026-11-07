import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
MIDPOINTS = (LEVELS[:-1] + LEVELS[1:]) / 2
FORMAT = "jalapenos-mxfp4-v1"


def quantize(values):
    """OCP MXFP4, 32 values per E8M0 scale, with round-to-nearest-even."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 32:
        raise ValueError("expected a matrix whose row width is divisible by 32")
    if not np.isfinite(values).all():
        raise ValueError("cannot quantize non-finite weights")

    blocks = values.reshape(values.shape[0], -1, 32)
    maximum = np.max(np.abs(blocks), axis=-1)
    # frexp returns floor(log2(maximum)) + 1 for every nonzero block.
    exponent = np.clip(np.frexp(maximum)[1] - 3, -127, 127)
    exponent[maximum == 0] = -127
    scaled = np.ldexp(np.abs(blocks), -exponent[..., None])
    codes = np.searchsorted(MIDPOINTS, scaled, side="left").astype(np.uint8)
    midpoint = MIDPOINTS[np.minimum(codes, len(MIDPOINTS) - 1)]
    codes += ((codes & 1) != 0) & (scaled == midpoint)
    codes |= np.signbit(blocks).astype(np.uint8) << 3
    codes = codes.reshape(values.shape)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = (exponent + 127).astype(np.uint8)
    return np.ascontiguousarray(packed), np.ascontiguousarray(scales)


def dequantize(packed, scales):
    """Independent float32 reference used for checking the Rust path."""
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    magnitude = LEVELS[codes & 7]
    values = np.copysign(magnitude, np.where(codes & 8, np.float32(-1.0), np.float32(1.0)))
    return np.ldexp(values, np.repeat(scales.astype(np.int32) - 127, 32, axis=1)).astype(np.float32)


def convert(source):
    source = Path(source).resolve()
    destination = source.with_name(source.name + "-MXFP4")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors weights in {source}")

    # Publish the directory only after every shard and config was written.
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=source.parent))
    source_bytes = 0
    packed_bytes = 0
    try:
        for shard in shards:
            output = {}
            with safe_open(shard, framework="pt", device="cpu") as weights:
                for name in weights.keys():
                    tensor = weights.get_tensor(name)
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError(f"expected BF16 weights: {name}")
                    source_bytes += tensor.numel() * tensor.element_size()
                    if tensor.ndim == 2:
                        rows, width = tensor.shape
                        if width % 32:
                            raise ValueError(f"row width must be divisible by 32: {name}")
                        packed = np.empty((rows, width // 2), dtype=np.uint8)
                        scales = np.empty((rows, width // 32), dtype=np.uint8)
                        for start in range(0, rows, 512):
                            packed[start:start + 512], scales[start:start + 512] = quantize(
                                tensor[start:start + 512].float().numpy()
                            )
                        output[name + ".mxfp4_packed"] = packed
                        output[name + ".mxfp4_scales"] = scales
                        packed_bytes += packed.nbytes + scales.nbytes
                    elif tensor.ndim == 1:
                        # Preserve the BF16 bit pattern of normalization weights.
                        raw = tensor.view(torch.uint16).numpy().copy()
                        output[name] = raw
                        packed_bytes += raw.nbytes
                    else:
                        raise ValueError(f"unsupported tensor shape: {name}")
            # U16 is a portable way to save the retained BF16 normalization bits.
            save_file(output, temporary / shard.name, metadata={"format": FORMAT})
            print(f"converted {shard.name}", flush=True)
            del output

        for path in source.iterdir():
            if path.is_file() and (
                path.suffix in {".json", ".txt"} or path.name == "LICENSE"
            ) and path.name != "model.safetensors.index.json":
                shutil.copy2(path, temporary / path.name)
        config_path = temporary / "config.json"
        config = json.loads(config_path.read_text())
        config["quantization_config"] = {
            "quant_method": "mxfp4",
            "format": FORMAT,
            "block_size": 32,
            "weights_only": True,
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"{source_bytes:,} -> {packed_bytes:,} weight bytes ({source_bytes / packed_bytes:.2f}x smaller)")
    print(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="model names under weights/, or model directory paths")
    args = parser.parse_args()
    for model in args.models:
        path = Path(model)
        convert(path if path.is_dir() else ROOT / "weights" / model)


if __name__ == "__main__":
    main()
