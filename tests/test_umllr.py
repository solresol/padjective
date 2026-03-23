import pytest

from padjective.umllr import (
    BattleRecord,
    ProductRecord,
    _dedupe_rows_by_key,
    _derive_battles_from_records,
    _ensure_ablation_storage,
    _p_adic_distance,
    _run_fold,
    _select_coefficient,
    _select_default_prediction,
    _tag_order,
    snapshot_label,
    tag_order_run_key,
)


def test_p_adic_distance_basic_properties() -> None:
    assert _p_adic_distance(10, 10, 5) == 0.0
    assert _p_adic_distance(10, 15, 5) == 0.2
    assert _p_adic_distance(1, 6, 5) == 0.2
    assert _p_adic_distance(1, 2, 5) == 1.0


def test_select_coefficient_prefers_smallest_candidate() -> None:
    values = [0, 5]
    base = 5
    # Candidates 0 and 5 produce the same loss; the smaller is chosen.
    coefficient, candidates = _select_coefficient(values, base)
    assert coefficient == 0
    assert [candidate.candidate_value for candidate in candidates] == [0, 5]
    assert [candidate.was_selected for candidate in candidates] == [True, False]


def test_select_default_prediction_minimizes_padic_loss() -> None:
    # Products with no tags have encoded paths 5, 10, 15 (all divisible by 5)
    # Candidates are 1, 5, 10, 15
    # p-adic distance from 5 to {5, 10, 15}: 0 + 0.2 + 0.2 = 0.4
    # p-adic distance from 10 to {5, 10, 15}: 0.2 + 0 + 0.2 = 0.4
    # p-adic distance from 15 to {5, 10, 15}: 0.2 + 0.2 + 0 = 0.4
    # p-adic distance from 1 to {5, 10, 15}: 1 + 1 + 1 = 3.0
    # Tie between 5, 10, 15 - should pick smallest (5)
    no_tag_values = [5, 10, 15]
    candidate_values = [1, 5, 10, 15]
    base = 5
    assert _select_default_prediction(no_tag_values, candidate_values, base) == 5


def test_select_default_prediction_empty_no_tag_values() -> None:
    # When no products have missing tags, fall back to most common
    no_tag_values = []
    candidate_values = [1, 5, 5, 10]  # 5 is most common
    base = 5
    assert _select_default_prediction(no_tag_values, candidate_values, base) == 5


def test_select_default_prediction_picks_best_for_ultrametric() -> None:
    # Test case where one candidate is clearly better due to ultrametricity
    # Products with no tags: 25, 50 (both divisible by 25)
    # p-adic distance from 25 to {25, 50}: 0 + 0.04 = 0.04
    # p-adic distance from 50 to {25, 50}: 0.04 + 0 = 0.04
    # p-adic distance from 1 to {25, 50}: 1 + 1 = 2.0
    # Tie: pick smallest (25)
    no_tag_values = [25, 50]
    candidate_values = [1, 25, 50]
    base = 5
    assert _select_default_prediction(no_tag_values, candidate_values, base) == 25


def test_run_fold_returns_coefficients_and_predictions() -> None:
    records = [
        ProductRecord(product_id=1, tags=["ALPHA", "BETA"], encoded_path=11, cv_fold=0),
        ProductRecord(product_id=2, tags=["ALPHA"], encoded_path=4, cv_fold=1),
        ProductRecord(product_id=3, tags=["BETA"], encoded_path=9, cv_fold=0),
    ]
    battles = [BattleRecord(winner_tag="ALPHA", loser_tag="BETA", cv_fold=0)]

    fold_result = _run_fold(1, records, battles, base=5)

    assert fold_result.cv_fold == 1
    assert [coeff.tag for coeff in fold_result.coefficients] == ["ALPHA", "BETA"]
    assert fold_result.coefficients[0].coefficient == 11
    assert fold_result.coefficients[1].coefficient == 0

    assert len(fold_result.predictions) == 1
    prediction = fold_result.predictions[0]
    assert prediction.product_id == 2
    assert prediction.predicted_value == 11
    assert prediction.true_value == 4
    assert prediction.loss == 1.0
    assert fold_result.loss == 1.0


def test_tag_order_battle_strategy_excludes_holdout_fold() -> None:
    training = [
        ProductRecord(product_id=1, title="ALPHA BETA", tags=["ALPHA", "BETA"], encoded_path=11, cv_fold=0, taxonomy_id="T1", taxonomy_depth=2),
        ProductRecord(product_id=2, title="BETA GAMMA", tags=["BETA", "GAMMA"], encoded_path=12, cv_fold=0, taxonomy_id="T2", taxonomy_depth=2),
    ]
    battles = [
        BattleRecord(winner_tag="BETA", loser_tag="ALPHA", cv_fold=0),
        BattleRecord(winner_tag="ALPHA", loser_tag="GAMMA", cv_fold=1),
    ]

    order = _tag_order(training, battles, 1, strategy="battle_elo")

    assert order[0] == "BETA"
    assert order.index("GAMMA") < order.index("ALPHA")


def test_tag_order_random_strategy_is_seeded() -> None:
    training = [
        ProductRecord(product_id=1, title="A B C", tags=["ALPHA", "BETA", "GAMMA"], encoded_path=11, cv_fold=0, taxonomy_id="T1", taxonomy_depth=3),
    ]
    battles: list[BattleRecord] = []

    first = _tag_order(training, battles, 0, strategy="random", seed=7)
    second = _tag_order(training, battles, 0, strategy="random", seed=7)
    third = _tag_order(training, battles, 0, strategy="random", seed=13)

    assert first == second
    assert first != third


def test_tag_order_mean_title_position_prefers_later_tags() -> None:
    training = [
        ProductRecord(product_id=1, title="ALPHA BETA", tags=["ALPHA", "BETA"], encoded_path=11, cv_fold=0, taxonomy_id="T1", taxonomy_depth=2),
        ProductRecord(product_id=2, title="ALPHA GAMMA", tags=["ALPHA", "GAMMA"], encoded_path=12, cv_fold=0, taxonomy_id="T2", taxonomy_depth=2),
    ]

    order = _tag_order(training, [], 0, strategy="mean_title_position")

    assert order[0] in {"BETA", "GAMMA"}
    assert order[-1] == "ALPHA"


def test_tag_order_taxonomy_association_prefers_more_peaked_tags() -> None:
    training = [
        ProductRecord(product_id=1, title="ALPHA", tags=["ALPHA", "BETA"], encoded_path=11, cv_fold=0, taxonomy_id="T1", taxonomy_depth=2),
        ProductRecord(product_id=2, title="ALPHA", tags=["ALPHA"], encoded_path=12, cv_fold=0, taxonomy_id="T2", taxonomy_depth=2),
        ProductRecord(product_id=3, title="BETA", tags=["BETA"], encoded_path=13, cv_fold=0, taxonomy_id="T1", taxonomy_depth=2),
    ]

    order = _tag_order(training, [], 0, strategy="taxonomy_association")

    assert order[0] == "BETA"


def test_tag_order_run_key_includes_snapshot_namespace() -> None:
    assert snapshot_label(None) == "live"
    assert snapshot_label("paper") == "paper"
    assert tag_order_run_key("battle_elo") == "live::battle_elo"
    assert tag_order_run_key("random", 7, snapshot_ref="paper") == "paper::random_seed_7"


def test_dedupe_rows_by_key_warns_and_preserves_first_row() -> None:
    rows = [
        (0, "LEAD", 11, 0),
        (0, "LEAD", 17, 1),
        (1, "LEAD", 19, 0),
    ]

    with pytest.warns(RuntimeWarning, match="duplicate UMLLR coefficient"):
        deduped = _dedupe_rows_by_key(
            rows,
            key_fn=lambda row: (row[0], row[1]),
            description="UMLLR coefficient",
        )

    assert deduped == [
        (0, "LEAD", 11, 0),
        (1, "LEAD", 19, 0),
    ]


def test_derive_battles_from_records_uses_record_folds() -> None:
    records = [
        ProductRecord(
            product_id=1,
            title="ALPHA BETA",
            tags=["ALPHA", "BETA"],
            encoded_path=11,
            cv_fold=3,
            taxonomy_id="T1",
            taxonomy_depth=2,
        ),
        ProductRecord(
            product_id=2,
            title="GAMMA",
            tags=["GAMMA"],
            encoded_path=12,
            cv_fold=4,
            taxonomy_id="T2",
            taxonomy_depth=1,
        ),
    ]

    battles = _derive_battles_from_records(records)

    assert battles == [
        BattleRecord(winner_tag="BETA", loser_tag="ALPHA", cv_fold=3),
    ]


class _AblationStorageCursor:
    def __init__(self, statements: list[tuple[str, object]]) -> None:
        self._statements = statements
        self._fetchone = None

    def __enter__(self) -> "_AblationStorageCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query, params=None) -> None:
        text = str(query)
        self._statements.append((text, params))
        if text == "SELECT CURRENT_USER":
            self._fetchone = ("padjective",)
        elif "ALTER TABLE" in text or "CREATE INDEX" in text:
            raise AssertionError(f"unexpected owner-only DDL: {text}")
        else:
            self._fetchone = None

    def fetchone(self):
        return self._fetchone


class _AblationStorageConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object]] = []
        self.commit_count = 0

    def cursor(self) -> _AblationStorageCursor:
        return _AblationStorageCursor(self.statements)

    def commit(self) -> None:
        self.commit_count += 1


def test_ensure_ablation_storage_skips_owner_only_ddl_when_tables_are_current(monkeypatch) -> None:
    conn = _AblationStorageConnection()

    monkeypatch.setattr("padjective.umllr.db.ensure_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("padjective.umllr.db.ensure_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "padjective.umllr._table_columns",
        lambda _conn, _schema, _table: {"run_key", "snapshot_ref"},
    )
    monkeypatch.setattr("padjective.umllr._table_owner", lambda *_args, **_kwargs: "gregb")
    monkeypatch.setattr("padjective.umllr._index_exists", lambda *_args, **_kwargs: True)

    _ensure_ablation_storage(conn, "padjective")

    executed = "\n".join(text for text, _params in conn.statements)
    assert "SELECT CURRENT_USER" in executed
    assert "UPDATE" in executed
    assert conn.commit_count == 1


def test_ensure_ablation_storage_raises_when_non_owner_needs_schema_upgrade(monkeypatch) -> None:
    conn = _AblationStorageConnection()

    monkeypatch.setattr("padjective.umllr.db.ensure_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("padjective.umllr.db.ensure_table", lambda *_args, **_kwargs: None)

    def _fake_columns(_conn, _schema, table):
        if table == "umllr_order_ablation_fold_metrics":
            return {"run_key"}
        return {"run_key", "snapshot_ref"}

    monkeypatch.setattr("padjective.umllr._table_columns", _fake_columns)
    monkeypatch.setattr("padjective.umllr._table_owner", lambda *_args, **_kwargs: "gregb")
    monkeypatch.setattr("padjective.umllr._index_exists", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="does not own the table"):
        _ensure_ablation_storage(conn, "padjective")
