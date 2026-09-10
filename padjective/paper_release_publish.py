"""Seal and publish the additive 10 September archive after complete validation.

Preparation and publication are separate explicit operations. Publication adds
only a previously absent dated directory and tag; it never deletes Hub files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from .paper_followup_release import README, REFERENCE_COMMIT, REFERENCE_TAG_COMMIT


REPO = "gregb/product-taxonomy-bench"
TAG = "paper-submission-2026-09-10"
PREFIX = "submission/2026-09-10/"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files(root):
    return {str(path.relative_to(root)): path for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and ".cache" not in path.parts}


def check_report(report, suite, manifest_hash, expected_rows):
    assert report["status"] == "passed" and report["full_grid"] is True
    assert report["context"]["suite"] == suite and report["context"]["manifest_sha256"] == manifest_hash
    assert report["cases"] == len(report["results"]) == expected_rows
    assert set(report["context"]["folds"]) == set(range(5))
    if suite == "ensembles":
        assert report["context"]["members"] == 243
        assert report["ensemble_rows"] == len(report["ensemble_checks"]) == 945
        assert len({(r["fold"], r["seed"]) for r in report["results"]}) == 1215
        assert {r["status"] for r in report["results"]} == {"coordinate_optimum"}
        assert len({(r["fold"], r["roster"], r["members"]) for r in report["ensemble_checks"]}) == 945
    else:
        assert len({r["run_id"] for r in report["results"]}) == 65
        assert Counter(r["status"] for r in report["results"]) == {
            "schedule_complete": 45, "rank_obstruction": 10, "inclusion_obstruction": 5, "time_limit": 5}


def copy_paper(repository, root):
    """Copy committed scientific source only, retaining relative bibliography paths."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    paths = subprocess.check_output(["git", "ls-files", "padjective/padic-journal"], cwd=repository, text=True).splitlines()
    selected = [name for name in paths if (Path(name).suffix in {
        ".tex", ".rty", ".bst", ".dot", ".eps", ".png", ".py", ".json", ".csv"}
        or Path(name).name == "Makefile" or name.endswith("/data/README.md"))
        and Path(name).name != "cover-letter.tex"]
    selected += ["bibliography.bib", "output/pdf/padjective-padic-journal.pdf"]
    for name in selected:
        # Reject uncommitted changes rather than freezing untraceable bytes.
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repository)
        assert content == (repository / name).read_bytes(), name
        destination = root / "paper-evidence" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (root / "paper-evidence/README.md").write_text(
        "# Frozen manuscript and figures\n\n"
        f"Papers source revision: `{commit}`.\n\n"
        "This directory preserves the relative source/bibliography layout.\n"
        "Install LaTeX (including REVTeX4), latexmk, Graphviz and Ghostscript.\n"
        "From `padjective/padic-journal/`, with the release's Python environment:\n\n"
        "```sh\n../../../.venv/bin/python -m unittest -q test_ensemble_assets test_published_methods_assets\n"
        "make PYTHON=../../../.venv/bin/python padjective-padic.pdf historical-comparator-diagnostics.pdf\n```\n\n"
        "Adjust the environment path if you installed it elsewhere. The canonical\n"
        "PDF is also under `output/pdf/`. Generators reproduce plots and tables\n"
        "from the original immutable aggregate evidence; `followup_replication.py`\n"
        "at the release root independently refits the models. PDF binary hashes\n"
        "can vary with TeX engine, fonts and build timestamps. Author annotations,\n"
        "submission correspondence and private per-product evidence are excluded.\n")
    return commit


def seal(root, ensembles, published, paper):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["validation_status"] == "pending", "Already sealed"
    for name, digest in manifest["sha256"].items():
        assert sha256(root/name) == digest, name
    manifest_hash = sha256(manifest_path)
    ensemble_report, published_report = [json.loads(path.read_text()) for path in (ensembles, published)]
    check_report(ensemble_report, "ensembles", manifest_hash, 1215)
    check_report(published_report, "published", manifest_hash, 65)
    directory = root / "validation"
    directory.mkdir(exist_ok=False)
    shutil.copy2(manifest_path, directory / "input-manifest.json")
    shutil.copy2(ensembles, directory / "ensembles.json")
    shutil.copy2(published, directory / "published.json")
    paper_commit = copy_paper(paper, root)
    (root / "README.md").write_text(README)
    manifest.update(validation_status="passed", validation_input_manifest_sha256=manifest_hash,
                    manuscript_source_commit=paper_commit,
                    validation=dict(refitted_coordinate_models=1215, reconstructed_ensembles=945,
                                    refitted_zubarev_models=45, reproduced_mihara_cases=20,
                                    timing_counts_not_bitwise_targets=True),
                    sealing_source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    manifest["sha256"] = {name: sha256(path) for name, path in release_files(root).items() if name != "manifest.json"}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(event="release_sealed", files=len(manifest["sha256"])+1,
                         manifest_sha256=sha256(manifest_path), paper_commit=paper_commit)), flush=True)


def publish(root):
    from huggingface_hub import CommitOperationAdd, HfApi

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["validation_status"] == "passed"
    files = release_files(root)
    assert set(files) == set(manifest["sha256"]) | {"manifest.json"}
    for name, digest in manifest["sha256"].items():
        assert sha256(files[name]) == digest, name
    api = HfApi()
    assert api.whoami()["name"] == "gregb"
    refs = api.list_repo_refs(REPO, repo_type="dataset")
    tags = {r.name: r.target_commit for r in refs.tags}
    assert tags["paper-submission-2026-09-06"] == REFERENCE_TAG_COMMIT
    assert TAG not in tags, "Tag already exists; inspect before retrying"
    head = api.repo_info(REPO, repo_type="dataset").sha
    existing = api.list_repo_files(REPO, repo_type="dataset", revision=head)
    assert not any(name.startswith(PREFIX) for name in existing), "Dated directory exists; inspect before retrying"
    # Exact additive manifest, one commit, optimistic concurrency protection.
    result = api.create_commit(repo_id=REPO, repo_type="dataset", parent_commit=head,
        operations=[CommitOperationAdd(path_in_repo=PREFIX+name, path_or_fileobj=str(path)) for name, path in files.items()],
        commit_message="Freeze published-method and 243-member ensemble replication release (2026-09-10)")
    commit = result.oid
    print(json.dumps(dict(event="uploaded", commit=commit)), flush=True)
    # Verify public bytes by immutable commit before creating the dated tag.
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{PREFIX}"
    for name, path in files.items():
        content = urllib.request.urlopen(base+name, timeout=90).read()
        assert hashlib.sha256(content).hexdigest() == sha256(path), name
    # Verify every pre-existing path retains its immutable content identifier.
    before = {r.path: (r.blob_id, str(getattr(r, "lfs", None))) for r in api.list_repo_tree(REPO, repo_type="dataset", revision=head, recursive=True) if hasattr(r, "blob_id")}
    after = {r.path: (r.blob_id, str(getattr(r, "lfs", None))) for r in api.list_repo_tree(REPO, repo_type="dataset", revision=commit, recursive=True) if hasattr(r, "blob_id")}
    assert all(after.get(name) == value for name, value in before.items())
    api.create_tag(REPO, repo_type="dataset", tag=TAG, revision=commit,
                   tag_message="Frozen validated public-matrix replication of the September 8 and 10 paper experiments")
    refs = {r.name: r.target_commit for r in api.list_repo_refs(REPO, repo_type="dataset").tags}
    assert refs[TAG] == commit and refs["paper-submission-2026-09-06"] == REFERENCE_TAG_COMMIT
    receipt = dict(status="published_and_verified", repository=REPO, revision=commit, tag=TAG,
                   parent_commit=head, previous_reference_commit=REFERENCE_COMMIT,
                   preserved_previous_files=len(before), verified_new_files=len(files),
                   manifest_sha256=sha256(root/"manifest.json"),
                   url=f"https://huggingface.co/datasets/{REPO}/tree/{TAG}/{PREFIX.rstrip('/')}")
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--ensembles", type=Path)
    parser.add_argument("--published", type=Path)
    parser.add_argument("--paper-repo", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.publish:
        if not args.receipt or args.receipt.exists():
            parser.error("A new receipt path outside the release is required")
        assert not args.receipt.resolve().is_relative_to(args.release_dir.resolve())
        receipt = publish(args.release_dir)
        args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    else:
        if not all((args.ensembles, args.published, args.paper_repo)):
            parser.error("Sealing requires both validation reports and the paper checkout")
        seal(args.release_dir, args.ensembles, args.published, args.paper_repo)


if __name__ == "__main__":
    main()
