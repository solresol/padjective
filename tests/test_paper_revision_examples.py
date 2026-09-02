from __future__ import annotations

import pytest

from padjective.benchmark_runtime import ProductRecord
from padjective.paper_revision_examples import (
    decode_base_digits,
    sample_three_product_cycles,
    valuation,
)


def _record(product_id: int, tags: list[str]) -> ProductRecord:
    return ProductRecord(
        product_id=product_id,
        product_key=f"product-{product_id}",
        tags=tags,
        encoded_path=product_id + 1,
        cv_fold=product_id % 2,
        taxonomy_id=f"taxonomy-{product_id}",
        taxonomy_depth=2,
        title_tag_positions=(),
    )


def test_valuation_and_digit_display() -> None:
    assert valuation(25, prime_base=5) == 2
    assert valuation(-10, prime_base=5) == 1
    assert decode_base_digits(1 + 2 * 5 + 3 * 25, prime_base=5) == (1, 2, 3)
    with pytest.raises(ValueError, match="zero"):
        valuation(0, prime_base=5)


def test_cycle_sampler_detects_closed_three_product_pattern() -> None:
    records = [
        _record(0, ["a", "c"]),
        _record(1, ["a", "b"]),
        _record(2, ["b", "c"]),
    ]

    result = sample_three_product_cycles(records, draws=100, seed=42)

    assert result.valid_chains == 100
    assert result.closed_chains == 100
    assert result.closure_rate == pytest.approx(1.0)
    assert result.witness is not None
    assert set(result.witness.cycle_tags) == {"a", "b", "c"}


def test_cycle_sampler_rejects_dataset_without_recurrent_pairs() -> None:
    with pytest.raises(ValueError, match="two recurrent tags"):
        sample_three_product_cycles([_record(0, ["a", "b"])], draws=10)
