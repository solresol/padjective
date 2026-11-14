"""Seal and publish the additive 10 September archive after complete validation.

Preparation and publication are separate explicit operations. Publication adds
only a previously absent dated directory and tag; it never deletes Hub files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
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


def paper_source_files(repository):
    paths = subprocess.check_output(["git", "ls-files", "padjective/padic-journal"], cwd=repository, text=True).splitlines()
    selected = [name for name in paths if (Path(name).suffix in {
        ".tex", ".rty", ".bst", ".dot", ".eps", ".png", ".py", ".json", ".csv"}
        or Path(name).name == "Makefile" or name.endswith("/data/README.md"))
        and Path(name).name != "cover-letter.tex"]
    selected += ["bibliography.bib", "output/pdf/padjective-padic-journal.pdf"]
    return selected


def copy_paper(repository, root):
    """Copy committed scientific source only, retaining relative bibliography paths."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    selected = paper_source_files(repository)
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
    # This short independent check has no model-selection feedback path.
    from .followup_capacity_replication import verify_capacity
    sys.path.insert(0, str(root.resolve()))
    capacity = verify_capacity(root)
    (directory / "capacity.json").write_text(json.dumps(capacity, indent=2)+"\n")
    shutil.copy2(Path(__file__).with_name("followup_capacity_replication.py"), root / "followup_capacity_replication.py")
    paper_commit = copy_paper(paper, root)
    (root / "README.md").write_text(README)
    expected_files = set(manifest["sha256"]) | {
        "manifest.json", "followup_capacity_replication.py", "validation/input-manifest.json",
        "validation/ensembles.json", "validation/published.json", "validation/capacity.json",
        "paper-evidence/README.md"} | {"paper-evidence/"+name for name in paper_source_files(paper)}
    actual_files = set(release_files(root))
    assert actual_files == expected_files, dict(unexpected=sorted(actual_files-expected_files), missing=sorted(expected_files-actual_files))
    manifest.update(validation_status="passed", validation_input_manifest_sha256=manifest_hash,
                    manuscript_source_commit=paper_commit,
                    validation=dict(refitted_coordinate_models=1215, reconstructed_ensembles=945,
                                    refitted_zubarev_models=45, reproduced_mihara_cases=20,
                                    recomputed_posthoc_capacity_bounds=15,
                                    timing_counts_not_bitwise_targets=True),
                    sealing_source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    manifest["sha256"] = {name: sha256(path) for name, path in release_files(root).items() if name != "manifest.json"}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(event="release_sealed", files=len(manifest["sha256"])+1,
                         manifest_sha256=sha256(manifest_path), paper_commit=paper_commit)), flush=True)


def checked_files(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["validation_status"] == "passed"
    files = release_files(root)
    assert set(files) == set(manifest["sha256"]) | {"manifest.json"}
    for name, digest in manifest["sha256"].items():
        assert sha256(files[name]) == digest, name
    return files


def check_storage_rules(previous, current, new_paths):
    """Hub may append exact new-file LFS rules; never allow old/global rewrites."""
    assert current.startswith(previous), "Existing storage rules changed"
    additions = current[len(previous):].splitlines()
    paths = []
    for line in additions:
        if not line.strip():
            continue
        fields = line.split()
        assert len(fields) == 5 and fields[1:] == ["filter=lfs", "diff=lfs", "merge=lfs", "-text"]
        path = fields[0]
        assert path.startswith(PREFIX) and path in new_paths
        assert not any(c in path for c in "*?[]\\"), "Wildcard storage rule"
        paths.append(path)
    assert len(paths) == len(set(paths))
    return paths


def verify_and_tag(root, commit, previous_revision):
    from huggingface_hub import HfApi

    files = checked_files(root)
    api = HfApi()
    assert api.whoami()["name"] == "gregb"
    refs = {r.name: r.target_commit for r in api.list_repo_refs(REPO, repo_type="dataset").tags}
    assert refs["paper-submission-2026-09-06"] == REFERENCE_TAG_COMMIT
    # The refs endpoint returns the annotated tag-object OID, not its peeled
    # content commit. Resolve via repo_info before comparing commit identities.
    assert api.repo_info(REPO, repo_type="dataset", revision="paper-submission-2026-09-06").sha == REFERENCE_COMMIT
    if TAG in refs:
        assert api.repo_info(REPO, repo_type="dataset", revision=TAG).sha == commit, "Never move an existing tag"
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{PREFIX}"

    def check_public(item):
        name, path = item
        content = urllib.request.urlopen(base+name, timeout=90).read()
        assert hashlib.sha256(content).hexdigest() == sha256(path), name

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(check_public, files.items()))
    print(json.dumps(dict(event="public_bytes_verified", files=len(files))), flush=True)
    before = {r.path: (r.blob_id, str(getattr(r, "lfs", None))) for r in api.list_repo_tree(REPO, repo_type="dataset", revision=previous_revision, recursive=True) if hasattr(r, "blob_id")}
    after = {r.path: (r.blob_id, str(getattr(r, "lfs", None))) for r in api.list_repo_tree(REPO, repo_type="dataset", revision=commit, recursive=True) if hasattr(r, "blob_id")}
    assert set(after)-set(before) == {PREFIX+name for name in files}
    assert not set(before)-set(after)
    changes = {name for name, value in before.items() if after[name] != value}
    assert changes <= {".gitattributes"}, sorted(changes)
    storage_rules = []
    if changes:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/"
        old = urllib.request.urlopen(url+previous_revision+"/.gitattributes", timeout=60).read().decode()
        new = urllib.request.urlopen(url+commit+"/.gitattributes", timeout=60).read().decode()
        storage_rules = check_storage_rules(old, new, {PREFIX+name for name in files})
    if TAG not in refs:
        api.create_tag(REPO, repo_type="dataset", tag=TAG, revision=commit,
                       tag_message="Frozen validated public-matrix replication of the September 8 and 10 paper experiments")
    refs = {r.name: r.target_commit for r in api.list_repo_refs(REPO, repo_type="dataset").tags}
    assert api.repo_info(REPO, repo_type="dataset", revision=TAG).sha == commit
    assert refs["paper-submission-2026-09-06"] == REFERENCE_TAG_COMMIT
    receipt = dict(status="published_and_verified", repository=REPO, revision=commit, tag=TAG,
                   tag_object=refs[TAG], previous_tag_object=REFERENCE_TAG_COMMIT,
                   parent_commit=previous_revision, previous_reference_commit=REFERENCE_COMMIT,
                   preserved_previous_files=len(before)-len(changes), verified_new_files=len(files),
                   appended_new_file_lfs_rules=storage_rules,
                   manifest_sha256=sha256(root/"manifest.json"),
                   url=f"https://huggingface.co/datasets/{REPO}/tree/{TAG}/{PREFIX.rstrip('/')}")
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def publish(root):
    from huggingface_hub import CommitOperationAdd, HfApi

    files = checked_files(root)
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
    return verify_and_tag(root, commit, head)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--ensembles", type=Path)
    parser.add_argument("--published", type=Path)
    parser.add_argument("--paper-repo", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify-upload", help="Verify/tag this observed upload commit without uploading again")
    parser.add_argument("--previous-revision", help="Observed parent revision for --verify-upload")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.publish or args.verify_upload:
        if args.publish and args.verify_upload:
            parser.error("Choose new publication or verification of an existing upload")
        if not args.receipt or args.receipt.exists():
            parser.error("A new receipt path outside the release is required")
        assert not args.receipt.resolve().is_relative_to(args.release_dir.resolve())
        if args.verify_upload:
            if not args.previous_revision:
                parser.error("Existing upload verification requires its observed parent revision")
            receipt = verify_and_tag(args.release_dir, args.verify_upload, args.previous_revision)
        else:
            receipt = publish(args.release_dir)
        args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    else:
        if not all((args.ensembles, args.published, args.paper_repo)):
            parser.error("Sealing requires both validation reports and the paper checkout")
        seal(args.release_dir, args.ensembles, args.published, args.paper_repo)


if __name__ == "__main__":
    main()
