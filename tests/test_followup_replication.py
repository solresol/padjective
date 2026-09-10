import ast
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pytest

from padjective.paper_followup_release import HELPERS, build_algorithms, extract_helpers
from padjective.paper_release_publish import check_report, release_files


SOURCE = Path(__file__).resolve().parents[1] / "padjective"


@pytest.fixture
def standalone(tmp_path, monkeypatch):
    build_algorithms(SOURCE, tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    names = [name for name in sys.modules if name == "followup_replication" or name == "algorithms" or name.startswith("algorithms.")]
    for name in names:
        monkeypatch.delitem(sys.modules, name)
    runtime = importlib.import_module("followup_replication")
    yield tmp_path, runtime
    for name in list(sys.modules):
        if name == "followup_replication" or name == "algorithms" or name.startswith("algorithms."):
            sys.modules.pop(name)


def test_numerical_sources_are_byte_identical(standalone):
    root, _ = standalone
    for name in ("published_mihara.py", "published_zubarev.py", "randomised_linear.py"):
        assert (root / "algorithms" / name).read_bytes() == (SOURCE / name).read_bytes()
    for path in [root / "followup_replication.py", *(root / "algorithms").glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not any(name.name.startswith(("psycopg", "padjective")) for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in {"db", "data_access"} and not (node.module or "").startswith("psycopg")


def test_extracted_audits_have_identical_asts():
    copied = {node.name: node for node in ast.parse(extract_helpers(SOURCE)).body if isinstance(node, ast.FunctionDef)}
    for filename, names in HELPERS.items():
        original = {node.name: node for node in ast.parse((SOURCE / filename).read_text()).body if isinstance(node, ast.FunctionDef)}
        for name in names:
            assert ast.dump(original[name], include_attributes=False) == ast.dump(copied[name], include_attributes=False)


def write_rows(path, rows):
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row)+"\n")


def test_loader_preserves_tag_rank_and_stored_folds(standalone):
    root, runtime = standalone
    directory = root / "reference/paper"
    directory.mkdir(parents=True)
    write_rows(directory / "tags.jsonl.gz", [dict(tag_id="a", tag_rank=2), dict(tag_id="z", tag_rank=1)])
    write_rows(directory / "products-00000.jsonl.gz", [
        dict(product_id_hash="second", tag_count=2, tag_features=[dict(tag_id="z"), dict(tag_id="a")], taxonomy_path="2.3", cv_fold=4),
        dict(product_id_hash="first", tag_count=0, tag_features=[dict(tag_id="a")], taxonomy_path="1", cv_fold=0)])
    matrix, targets, folds, names, digest = runtime.load_matrix(root)
    assert names == ("z", "a")
    assert matrix.toarray().tolist() == [[0, 1], [1, 1]]
    assert targets.tolist() == [1, 215] and folds.tolist() == [0, 4]
    ordered = [["first", 0, 1, ["a"]], ["second", 4, 215, ["a", "z"]]]
    expected = b"".join(json.dumps(row, separators=(",", ":")).encode()+b"\n" for row in ordered)
    assert digest == hashlib.sha256(expected).hexdigest()


def test_loader_rejects_duplicate_features(standalone):
    root, runtime = standalone
    directory = root / "reference/paper"
    directory.mkdir(parents=True)
    write_rows(directory / "tags.jsonl.gz", [dict(tag_id="a", tag_rank=1)])
    write_rows(directory / "products-00000.jsonl.gz", [dict(product_id_hash="one", tag_count=2,
               tag_features=[dict(tag_id="a"), dict(tag_id="a")], taxonomy_path="1", cv_fold=0)])
    with pytest.raises(AssertionError):
        runtime.load_matrix(root)


def test_release_fails_closed_on_changed_file(standalone):
    root, runtime = standalone
    manifest = dict(python_version=platform.python_version(), versions={}, sha256={"README.md": hashlib.sha256(b"original").hexdigest()})
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "README.md").write_bytes(b"original")
    assert runtime.verify_release(root) == manifest
    (root / "README.md").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runtime.verify_release(root)


def test_standalone_exact_fit_and_consensus(standalone):
    _, runtime = standalone
    x = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y = np.array([1, 1, 2, 2])
    fit = runtime.fit_coordinates(x, y, p=3, precision=2, seed=42)
    assert fit.status == "coordinate_optimum" and fit.coefficients.tolist() == [1, 2]
    assert runtime.direct_certificate(runtime.sparse.csr_matrix(x), y, fit.coefficients, y.tolist(), 3, 2) == 2
    bank = np.array([[1, 4, 1], [2, 2, 5]])
    options = np.array([1, 2, 4, 5])
    checks = list(runtime.nested_consensus(bank, options, [1, 3], [np.arange(3)], p=3, precision=2))
    for _, size, predictions in checks:
        original, _ = runtime.consensus_predictions(bank[:, :size], options, p=3, precision=2)
        assert np.array_equal(predictions, original)


def test_sealing_rejects_smoke_runs_and_wrong_manifest():
    report = dict(status="passed", full_grid=False)
    with pytest.raises(AssertionError):
        check_report(report, "ensembles", "expected", 1215)
    report.update(full_grid=True, context=dict(suite="ensembles", manifest_sha256="different"))
    with pytest.raises(AssertionError):
        check_report(report, "ensembles", "expected", 1215)


def test_publication_inventory_excludes_bytecode_and_download_cache(tmp_path):
    for name in ("manifest.json", "algorithms/model.py", "algorithms/__pycache__/model.pyc", ".cache/huggingface/state"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content")
    assert set(release_files(tmp_path)) == {"manifest.json", "algorithms/model.py"}
