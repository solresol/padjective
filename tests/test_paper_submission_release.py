import sys
from types import ModuleType, SimpleNamespace

from padjective import benchmark_runtime, taxonomy_mihara_comparison
from padjective.paper_submission_release import anonymise_diagnostic, render_digitwise_runtime
from padjective.product_taxonomy_bench_notebook import render_notebook


def test_diagnostic_export_excludes_source_identity_and_preserves_tag_order():
    records = [SimpleNamespace(
        product_id=3456, title="private title", tags=["ZEBRA", "APPLE"],
        encoded_path=72, cv_fold=2, taxonomy_id="public-taxonomy", taxonomy_depth=2,
    )]
    rows = anonymise_diagnostic(records)
    assert rows[0]["tags"] == ["tag000002", "tag000001"]
    assert rows[0]["product_id"] == 0
    assert rows[0]["cv_fold"] == 2
    assert "title" not in rows[0]
    assert "private" not in str(rows) and "ZEBRA" not in str(rows)


def test_generated_digitwise_runtime_matches_source_without_database_imports(monkeypatch):
    source = render_digitwise_runtime()
    assert "psycopg" not in source and "db.get_connection" not in source
    module = ModuleType("standalone_digitwise_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "benchmark_runtime", benchmark_runtime)
    exec(compile(source, "digitwise_runtime.py", "exec"), module.__dict__)
    features = [(0,), (1,), (0,), (1,), (0,), (1,)]
    targets = [1, 3, 1, 3, 1, 3]
    kwargs = dict(p=5, precision=2, trials=4, seed=0)
    actual = module.fit_mihara_digitwise(features, targets, **kwargs)
    expected = taxonomy_mihara_comparison.fit_mihara_digitwise(features, targets, **kwargs)
    assert actual.coefficients == expected.coefficients
    assert actual.intercept == expected.intercept
    assert actual.accepted_prefix_digits == expected.accepted_prefix_digits


def test_notebook_pins_paper_configuration_and_distinct_primary_roster():
    source = "\n".join("".join(cell["source"]) for cell in render_notebook()["cells"])
    assert '"paper-submission-2026-09-06"' in source
    assert 'PRODUCT_TAXONOMY_BENCH_UNN_HIDDEN", "2000"' in source
    assert 'ACTIVE_PARAMS_EXCLUDED = {"pclr", "pcnn", "zubarev"}' in source
