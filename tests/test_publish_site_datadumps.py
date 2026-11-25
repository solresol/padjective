from pathlib import Path
import shutil
import stat
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/publish_site_datadumps.sh"
pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="requires rsync")


@pytest.fixture
def dirs(tmp_path):
    source = tmp_path / "build" / "datadumps"
    site = tmp_path / "vhosts" / "padjective.symmachus.org" / "htdocs"
    legacy = site / "datadumps"
    canonical = tmp_path / "vhosts" / "datadumps.ifost.org.au" / "htdocs" / "padjective"
    for path in (source, legacy, canonical):
        path.mkdir(parents=True)
    legacy.chmod(0o775)
    return source, canonical, legacy


def publish(source, canonical, legacy, *, check=True):
    return subprocess.run(
        ["bash", SCRIPT, source, f"{canonical}/", f"{legacy}/"],
        check=check, capture_output=True,
    )


def test_publish_single_copy_with_original_relative_links(dirs):
    source, canonical, legacy = dirs
    (source / "current.sql").write_text("new dump")
    (canonical / "historical.sql").write_text("retain historical export")
    (legacy / "current.sql").write_text("old duplicate")
    (legacy / "retired.sql").symlink_to("current.sql")
    publish(*dirs)

    assert (canonical / "current.sql").read_text() == "new dump"
    assert (legacy / "current.sql").is_symlink()
    assert (legacy / "current.sql").resolve() == canonical / "current.sql"
    assert (legacy / "current.sql").read_text() == "new dump"
    assert (canonical / "historical.sql").read_text() == "retain historical export"
    assert not (legacy / "historical.sql").exists()
    assert not (legacy / "retired.sql").is_symlink()
    assert stat.S_IMODE(legacy.stat().st_mode) == 0o775

    # The main site's --delete pass must not delete the compatibility directory.
    (source.parent / "index.html").write_text("new page")
    subprocess.run(
        ["rsync", "-az", "--delete", "--exclude=/datadumps/", f"{source.parent}/",
         f"{legacy.parent}/"],
        check=True,
    )
    assert (legacy.parent / "index.html").read_text() == "new page"
    assert (legacy / "current.sql").read_text() == "new dump"


def test_empty_build_does_not_clear_downloads(dirs):
    _, canonical, legacy = dirs
    (canonical / "current.sql").write_text("old dump")
    (legacy / "current.sql").write_text("old dump")
    publish(*dirs)
    assert (legacy / "current.sql").read_text() == "old dump"
    assert (canonical / "current.sql").read_text() == "old dump"


def test_invalid_build_does_not_publish_any_files(dirs):
    source, canonical, legacy = dirs
    (source / "valid.sql").write_text("new dump")
    (source / "link.sql").symlink_to("valid.sql")
    assert publish(*dirs, check=False).returncode != 0
    assert list(canonical.iterdir()) == list(legacy.iterdir()) == []


def test_failed_canonical_upload_leaves_legacy_files_alone(dirs):
    source, canonical, legacy = dirs
    (source / "current.sql").write_text("new dump")
    (legacy / "current.sql").write_text("old dump")
    assert publish(source, canonical / "missing" / "nested", legacy, check=False).returncode != 0
    assert not (legacy / "current.sql").is_symlink()
    assert (legacy / "current.sql").read_text() == "old dump"
