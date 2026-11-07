import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

spec = importlib.util.spec_from_file_location(
    "quantize_mxfp4", Path(__file__).resolve().parents[1] / "quantize_mxfp4.py"
)
mxfp4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mxfp4)


def test_all_codes_round_trip_with_signed_zero():
    values = np.tile(np.concatenate([mxfp4.LEVELS, -mxfp4.LEVELS]), 2)[None, :]
    packed, scales = mxfp4.quantize(values)
    np.testing.assert_array_equal(packed[0, :8], [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE])
    np.testing.assert_array_equal(scales, [[127]])
    actual = mxfp4.dequantize(packed, scales)
    np.testing.assert_array_equal(actual, values)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(values))


def test_midpoints_round_to_even_and_overflow_saturates():
    values = np.zeros((1, 32), dtype=np.float32)
    values[0, :8] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0]
    packed, scales = mxfp4.quantize(values)
    np.testing.assert_array_equal(scales, [[127]])
    np.testing.assert_array_equal(
        mxfp4.dequantize(packed, scales)[0, :8], [0, 1, 1, 2, 2, 4, 4, 6]
    )


def test_each_block_gets_its_own_scale():
    values = np.zeros((2, 64), dtype=np.float32)
    values[0, :32] = 6
    values[0, 32:] = -0.75
    packed, scales = mxfp4.quantize(values)
    np.testing.assert_array_equal(scales, [[127, 124], [0, 0]])
    np.testing.assert_array_equal(mxfp4.dequantize(packed, scales), values)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_non_finite_weights_are_rejected(invalid):
    values = np.zeros((1, 32), dtype=np.float32)
    values[0, 1] = invalid
    with pytest.raises(ValueError, match="non-finite"):
        mxfp4.quantize(values)


def test_invalid_row_width_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        mxfp4.quantize(np.zeros((1, 31)))


def test_converter_writes_packed_checkpoint_and_cleans_up_failure():
    with tempfile.TemporaryDirectory() as root:
        source = Path(root) / "tiny"
        source.mkdir()
        (source / "config.json").write_text('{}')
        save_file({
            "matrix": torch.full((2, 32), 6, dtype=torch.bfloat16),
            "norm": torch.ones(32, dtype=torch.bfloat16),
        }, source / "model.safetensors")
        destination = mxfp4.convert(source)
        with safe_open(destination / "model.safetensors", framework="numpy") as weights:
            assert weights.metadata()["format"] == mxfp4.FORMAT
            np.testing.assert_array_equal(weights.get_tensor("matrix.mxfp4_packed"), 0x77)
            np.testing.assert_array_equal(weights.get_tensor("matrix.mxfp4_scales"), 127)
            np.testing.assert_array_equal(weights.get_tensor("norm"), 0x3F80)
        with pytest.raises(FileExistsError):
            mxfp4.convert(source)
        broken = Path(root) / "broken"
        broken.mkdir()
        save_file({"matrix": torch.ones(2, 31, dtype=torch.bfloat16)}, broken / "model.safetensors")
        with pytest.raises(ValueError, match="divisible"):
            mxfp4.convert(broken)
        assert not list(Path(root).glob(".broken-MXFP4-*"))
        assert not (Path(root) / "broken-MXFP4").exists()
