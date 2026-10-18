from datetime import timezone
import os
from pathlib import Path
import subprocess
import sys

import pytest

import padjective.product_taxonomy_bench_publish as publish_mod
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


@pytest.mark.parametrize(
    "extra_args, expected_dataset_id, expected_repo_id",
    [
        ([], "gregb/shimp", None),
        (["--hf-repo-id", "gregb/shimp"], "gregb/shimp", "gregb/shimp"),
        (
            ["--hf-repo-id", "gregb/product-taxonomy-bench"],
            "gregb/product-taxonomy-bench", "gregb/product-taxonomy-bench",
        ),
        (["--hf-repo-id", "someone/custom"], "someone/custom", "someone/custom"),
        (
            ["--hf-repo-id", "someone/custom", "--dataset-id", "custom-card",
             "--pretty-name", "Custom dataset"],
            "custom-card", "someone/custom",
        ),
    ],
)
def test_publish_cli_metadata_and_upload_opt_in(
    monkeypatch, tmp_path, extra_args, expected_dataset_id, expected_repo_id
):
    calls = []
    monkeypatch.setattr(publish_mod, "publish", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv",
        ["publish", "--out-root", str(tmp_path), "--paper-as-of",
         "2026-02-11 19:15 UTC", *extra_args],
    )

    publish_mod.main()

    assert len(calls) == 1
    assert calls[0]["dataset_id"] == expected_dataset_id
    assert calls[0]["hf_repo_id"] == expected_repo_id
    expected_name = (
        "Custom dataset" if "--pretty-name" in extra_args
        else "SHIMP (Shopify Hierarchy Inference from Merchant Products)"
    )
    assert calls[0]["pretty_name"] == expected_name


@pytest.mark.parametrize("override", [None, "gregb/product-taxonomy-bench", "someone/custom"])
def test_weekly_publisher_target_default_and_override(tmp_path, override):
    # Stub commands: exercise the wrapper without a database, pull, or Hub upload.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "args.txt"
    for command, source in {
        "git": "#!/bin/bash\nexit 0\n",
        "uv": '#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n',
    }.items():
        stub = bin_dir / command
        stub.write_text(source)
        stub.chmod(0o755)
    env = os.environ.copy()
    env.pop("PADJECTIVE_HF_REPO_ID", None)
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "CAPTURE_ARGS": str(args_file),
        "PADJECTIVE_HF_OUT_ROOT": str(tmp_path / "exports"),
        "PADJECTIVE_PAPER_GENERATED_DIR": str(tmp_path / "generated"),
    })
    if override:
        env["PADJECTIVE_HF_REPO_ID"] = override
    wrapper = Path(__file__).resolve().parents[1] / "weekly_hf_publish.sh"

    subprocess.run(["bash", str(wrapper)], env=env, check=True)

    args = args_file.read_text().splitlines()
    assert args[args.index("--hf-repo-id") + 1] == (override or "gregb/shimp")

