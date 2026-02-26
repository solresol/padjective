from datetime import timezone

import pytest

from padjective.product_taxonomy_bench_publish import parse_latest_utc_timestamp_from_tex


def test_parse_latest_utc_timestamp_from_tex_returns_max() -> None:
    tex = (
        "Checked on 2026-02-05 10:00 UTC and later on 2026-02-11 19:15 UTC.\n"
        "Another earlier 2026-01-24 19:03 UTC."
    )
    dt = parse_latest_utc_timestamp_from_tex(tex)
    assert dt.isoformat() == "2026-02-11T19:15:00+00:00"
    assert dt.tzinfo == timezone.utc


def test_parse_latest_utc_timestamp_from_tex_raises_without_match() -> None:
    with pytest.raises(ValueError, match="No 'YYYY-MM-DD HH:MM UTC' timestamps"):
        parse_latest_utc_timestamp_from_tex("no timestamps here")

