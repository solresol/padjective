"""Stage the licence notices shared by SHIMP exports and replication packages."""

from pathlib import Path


RELEASE_LICENSE_FILES = (
    "LICENSE",
    "LICENSE-DATA",
    "LICENSE-CODE",
    "THIRD_PARTY_NOTICES.md",
)


def stage_release_licenses(out_root: Path) -> tuple[Path, ...]:
    source_root = Path(__file__).resolve().parent.parent
    contents = {name: (source_root / name).read_bytes() for name in RELEASE_LICENSE_FILES}
    out_root.mkdir(parents=True, exist_ok=True)
    destinations = []
    for name, content in contents.items():
        destination = out_root / name
        destination.write_bytes(content)
        destinations.append(destination)
    return tuple(destinations)
