import pytest

from padjective.umllr import (
    BattleRecord,
    DEFAULT_TAG_ORDER_STRATEGY,
    ProductRecord,
    FoldResult,
    Prediction,
    _acquire_session_lock,
    _dedupe_rows_by_key,
    _derive_battles_from_records,
    _ensure_ablation_storage,
    _p_adic_distance,
    _run_fold,
    _save_ablation_results,
    _select_coefficient,
    _select_default_prediction,
    _tag_order,
    process_database,
    snapshot_label,
    tag_order_run_key,
)


def test_default_tag_order_strategy_uses_taxonomy_association() -> None:
    assert DEFAULT_TAG_ORDER_STRATEGY == "taxonomy_association"


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

    fold_result = _run_fold(1, records, battles, base=5, tag_order_strategy="battle_elo")

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
    def __init__(self, statements: list[tuple[str, object]], *, allow_owner_ddl: bool = False) -> None:
        self._statements = statements
        self._fetchone = None
        self._allow_owner_ddl = allow_owner_ddl

    def __enter__(self) -> "_AblationStorageCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query, params=None) -> None:
        text = str(query)
        self._statements.append((text, params))
        if text == "SELECT CURRENT_USER":
            self._fetchone = ("padjective",)
        elif ("ALTER TABLE" in text or "CREATE INDEX" in text) and not self._allow_owner_ddl:
            raise AssertionError(f"unexpected owner-only DDL: {text}")
        else:
            self._fetchone = None

    def fetchone(self):
        return self._fetchone

    def executemany(self, query, params_seq) -> None:
        self._statements.append((str(query), list(params_seq)))


class _AblationStorageConnection:
    def __init__(self, *, allow_owner_ddl: bool = False) -> None:
        self.statements: list[tuple[str, object]] = []
        self.commit_count = 0
        self.closed = False
        self.allow_owner_ddl = allow_owner_ddl

    def cursor(self) -> _AblationStorageCursor:
        return _AblationStorageCursor(self.statements, allow_owner_ddl=self.allow_owner_ddl)

    def commit(self) -> None:
        self.commit_count += 1

    def close(self) -> None:
        self.closed = True


def test_ensure_ablation_storage_skips_owner_only_ddl_when_tables_are_current(monkeypatch) -> None:
    conn = _AblationStorageConnection()

    monkeypatch.setattr("padjective.umllr.db.ensure_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("padjective.umllr.db.ensure_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "padjective.umllr._table_columns",
        lambda _conn, _schema, _table: {"run_key", "snapshot_ref", "updated_at"},
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


def test_ensure_ablation_storage_adds_updated_at_when_owner_can_migrate(monkeypatch) -> None:
    conn = _AblationStorageConnection(allow_owner_ddl=True)

    monkeypatch.setattr("padjective.umllr.db.ensure_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("padjective.umllr.db.ensure_table", lambda *_args, **_kwargs: None)

    def _fake_columns(_conn, _schema, table):
        if table == "umllr_order_ablation_fold_metrics":
            return {"run_key", "snapshot_ref", "updated_at"}
        return {"run_key", "snapshot_ref"}

    monkeypatch.setattr("padjective.umllr._table_columns", _fake_columns)
    monkeypatch.setattr("padjective.umllr._table_owner", lambda *_args, **_kwargs: "padjective")
    monkeypatch.setattr("padjective.umllr._index_exists", lambda *_args, **_kwargs: True)

    _ensure_ablation_storage(conn, "padjective")

    executed = "\n".join(text for text, _params in conn.statements)
    assert "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()" in executed
    assert conn.commit_count == 1


def test_acquire_session_lock_uses_postgres_advisory_lock() -> None:
    conn = _AblationStorageConnection()

    _acquire_session_lock(conn, "padjective.process_database")

    assert conn.statements == [
        (
            "SELECT pg_advisory_lock(hashtext(%s), hashtext(%s))",
            ("padjective.umllr", "padjective.process_database"),
        )
    ]


def test_save_ablation_results_uses_upserts(monkeypatch) -> None:
    conn = _AblationStorageConnection()

    monkeypatch.setattr("padjective.umllr._ensure_ablation_storage", lambda *_args, **_kwargs: None)

    results = [
        FoldResult(
            cv_fold=0,
            coefficients=[],
            predictions=[
                Prediction(product_id=101, true_value=11, predicted_value=13, loss=0.2),
            ],
            loss=0.2,
            default_prediction=5,
            tag_debug=[],
            exact_accuracy=0.4,
            prefix1_accuracy=0.6,
            prefix2_accuracy=0.8,
            mean_shared_prefix_depth=1.5,
            mean_scoring_ops=3.0,
        ),
    ]

    _save_ablation_results(
        conn,
        "padjective",
        results,
        run_key="live::taxonomy_association",
        snapshot_ref="live",
        tag_order_strategy="taxonomy_association",
        tag_order_seed=None,
        prime_base=5,
        max_digit=3,
    )

    metrics_insert = next(
        text
        for text, _params in conn.statements
        if "INSERT INTO" in text and "umllr_order_ablation_fold_metrics" in text
    )
    predictions_insert = next(
        text
        for text, _params in conn.statements
        if "INSERT INTO" in text and "umllr_order_ablation_predictions" in text
    )

    assert "ON CONFLICT (run_key, cv_fold) DO UPDATE SET" in metrics_insert
    assert "ON CONFLICT (run_key, cv_fold, product_id) DO UPDATE SET" in predictions_insert
    assert conn.commit_count == 1


def test_process_database_acquires_lock_before_work(monkeypatch) -> None:
    conn = _AblationStorageConnection()

    monkeypatch.setattr("padjective.umllr.db.get_connection", lambda _dsn: conn)
    monkeypatch.setattr("padjective.umllr._ensure_storage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("padjective.umllr.calculate_cv_folds", lambda *_args, **_kwargs: {})

    process_database(dsn=None, schema="padjective")

    assert conn.statements[0] == (
        "SELECT pg_advisory_lock(hashtext(%s), hashtext(%s))",
        ("padjective.umllr", "padjective.process_database"),
    )
    assert conn.closed is True
