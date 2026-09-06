import gzip
import io

import pandas as pd
import pytest

from padjective import benchmark_runtime as runtime


@pytest.mark.parametrize("compressed", [False, True])
def test_http_jsonl_stream_handles_explicit_compression(monkeypatch, compressed):
    content = b'{"tag_id":"tag000001","tag_rank":1}\n'
    if compressed:
        content = gzip.compress(content)
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda request: io.BytesIO(content))
    url = "https://example.test/tags.jsonl" + (".gz?download=true" if compressed else "")
    result = runtime._load_json_url_df(url)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"tag_id": ["tag000001"], "tag_rank": [1]}))


def test_hf_file_listing_uses_selected_revision(monkeypatch):
    calls = []

    def fake_json(url, **kwargs):
        calls.append(url)
        return {"siblings": []}

    monkeypatch.setattr(runtime, "_load_json_url", fake_json)
    with pytest.raises(ValueError, match="Snapshot folder"):
        runtime.load_snapshot_tables_from_hf(
            dataset_id="gregb/product-taxonomy-bench", revision="release/test", snapshot="paper",
        )
    assert calls == ["https://huggingface.co/api/datasets/gregb/product-taxonomy-bench/revision/release%2Ftest"]
