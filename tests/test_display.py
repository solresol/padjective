import pandas as pd
from pathlib import Path

from padjective.display import generate_outputs


def _make_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"tag": ["A", "B", "C"], "score": [3, 2, 1], "component": [0, 0, 0]})


def test_generate_outputs_truncated(tmp_path, capsys):
    df = _make_dataframe()
    html = tmp_path / "out.html"
    img = tmp_path / "plot.png"
    generate_outputs(df, html, img, rows=2)
    out_lines = capsys.readouterr().out.strip().splitlines()
    # header plus two rows
    assert len(out_lines) == 3
    assert html.exists()
    assert img.exists()


def test_generate_outputs_all(tmp_path, capsys):
    df = _make_dataframe()
    html = tmp_path / "out_all.html"
    img = tmp_path / "plot_all.png"
    generate_outputs(df, html, img, rows=0)
    out_lines = capsys.readouterr().out.strip().splitlines()
    # header plus three rows
    assert len(out_lines) == 4

