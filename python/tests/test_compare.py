import pytest
import torch

from jalapenos_reference.compare import (
    compare,
    cosine_similarity,
    max_abs_diff,
    relative_error,
)

# Worked example: p is the reference, r is a slightly-off output.
P = torch.tensor([1.0, -2.0, 0.5, 3.0])
R = torch.tensor([1.01, -1.98, 0.5, 2.97])


def test_identical_vectors():
    scores = compare(P, P.clone())
    assert scores["max_abs"] == pytest.approx(0.0)
    assert scores["relative_err"] == pytest.approx(0.0)
    assert scores["cos_similarity"] == pytest.approx(1.0)


def test_worked_example():
    assert max_abs_diff(R, P) == pytest.approx(0.03, abs=1e-6)
    assert relative_error(R, P) == pytest.approx(0.0099119, abs=1e-6)
    assert cosine_similarity(R, P) == pytest.approx(0.9999861, abs=1e-6)


def test_double_size_same_direction():
    # Same direction, twice as long: cosine can't see it, relative error can.
    assert cosine_similarity(2 * P, P) == pytest.approx(1.0)
    assert relative_error(2 * P, P) == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare(torch.ones(3), torch.ones(4))


def test_zeroed_actual_raises():
    with pytest.raises(ValueError, match="actual"):
        compare(torch.zeros(4), P)


def test_zeroed_expected_raises():
    with pytest.raises(ValueError, match="expected"):
        compare(P, torch.zeros(4))
