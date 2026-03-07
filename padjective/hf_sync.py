"""Hugging Face Hub upload helpers (optional dependency).

This module intentionally keeps ``huggingface_hub`` as an optional dependency so
the core project can run without it. Publishing boxes can install it and use
``uv run -m padjective.product_taxonomy_bench_publish --hf-repo-id ...`` for automated uploads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


class HuggingFaceHubUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class UploadOptions:
    repo_id: str
    token: str | None = None
    repo_type: str = "dataset"
    commit_message: str = "Update dataset export"
    replace_folders: tuple[str, ...] = ("latest",)


def _require_huggingface_hub():
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise HuggingFaceHubUnavailable(
            "huggingface_hub is not installed. Install it with `uv add huggingface-hub` "
            "or run without --hf-repo-id."
        ) from exc
    return HfApi


def _delete_remote_folder(api, *, repo_id: str, repo_type: str, folder: str, token: str | None, commit_message: str) -> None:
    folder = folder.strip().strip("/")
    if not folder:
        return

    delete_folder = getattr(api, "delete_folder", None)
    if callable(delete_folder):  # pragma: no cover - depends on hub version
        try:
            delete_folder(
                repo_id=repo_id,
                path_in_repo=folder,
                repo_type=repo_type,
                token=token,
                commit_message=commit_message,
            )
        except Exception as exc:  # pragma: no cover - depends on hub responses
            message = str(exc).lower()
            if "entry not found" in message or "does not exist" in message:
                return
            raise
        return

    # Fallback: delete files individually.
    files: list[str] = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
    prefix = folder + "/"
    for path in files:
        if path.startswith(prefix):
            api.delete_file(
                repo_id=repo_id,
                path_in_repo=path,
                repo_type=repo_type,
                token=token,
                commit_message=commit_message,
            )


def upload_export_root(out_root: Path, options: UploadOptions) -> None:
    """Upload an export directory to the Hugging Face Hub dataset repo.

    Expected directory layout:
      - README.md
      - paper/
      - latest/
      - (optional) notebooks/
      - (optional) other snapshot folders (e.g. first1000/)
    """

    if not out_root.exists():
        raise FileNotFoundError(str(out_root))

    readme_path = out_root / "README.md"
    paper_dir = out_root / "paper"
    latest_dir = out_root / "latest"

    missing: list[str] = []
    if not readme_path.is_file():
        missing.append("README.md")
    if not paper_dir.is_dir():
        missing.append("paper/")
    if not latest_dir.is_dir():
        missing.append("latest/")
    if missing:
        raise ValueError(
            f"Export root {out_root} is missing required paths: {', '.join(missing)}"
        )

    snapshot_dirs = sorted(
        [
            path
            for path in out_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ],
        key=lambda path: path.name,
    )

    HfApi = _require_huggingface_hub()
    api = HfApi()

    api.create_repo(
        repo_id=options.repo_id,
        repo_type=options.repo_type,
        exist_ok=True,
        token=options.token,
    )

    for folder in options.replace_folders:
        _delete_remote_folder(
            api,
            repo_id=options.repo_id,
            repo_type=options.repo_type,
            folder=folder,
            token=options.token,
            commit_message=f"{options.commit_message} (replace {folder}/)",
        )

    # Root README
    api.upload_file(
        repo_id=options.repo_id,
        repo_type=options.repo_type,
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        token=options.token,
        commit_message=options.commit_message,
    )

    # Snapshot folders
    for snapshot_dir in snapshot_dirs:
        api.upload_folder(
            repo_id=options.repo_id,
            repo_type=options.repo_type,
            folder_path=str(snapshot_dir),
            path_in_repo=snapshot_dir.name,
            token=options.token,
            commit_message=options.commit_message,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload a product-taxonomy-bench export directory to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id (e.g. username/product-taxonomy-bench)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Export root directory containing README.md, paper/, and latest/.",
    )
    parser.add_argument(
        "--token",
        help="Optional Hugging Face token (otherwise uses huggingface_hub defaults/env).",
    )
    parser.add_argument(
        "--commit-message",
        default="Update dataset export",
        help="Commit message for the Hub.",
    )
    parser.add_argument(
        "--replace-folders",
        default="latest",
        help="Comma-separated list of folders to delete before upload (default: latest).",
    )
    args = parser.parse_args()

    replace_folders = tuple(
        part.strip().strip("/")
        for part in args.replace_folders.split(",")
        if part.strip().strip("/")
    )

    upload_export_root(
        args.out_root,
        UploadOptions(
            repo_id=args.repo_id,
            token=args.token,
            commit_message=args.commit_message,
            replace_folders=replace_folders,
        ),
    )


if __name__ == "__main__":
    main()
