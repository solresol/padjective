"""Recover stable numeric Shopify taxonomy paths from mixed path metadata.

``cantbuymelove.taxonomy.taxonomy_path`` has historically served two different
purposes: older rows contain numeric hierarchy keys (for example ``1.1.13.8``),
while newer classification runs may replace them with human-readable display
paths.  Padjective needs the numeric key for grouping and model evaluation, so
it keeps a separate, derived reconciliation table in its own Postgres schema.

The numeric key is recovered in decreasing order of authority:

1. a numeric value still present in the live taxonomy table;
2. a numeric value preserved in an immutable Padjective benchmark snapshot;
3. the Shopify taxonomy ID, after its root-code-to-number mapping has been
   established by either of the first two sources.

Conflicting evidence is treated as an error.  A pipeline should stop loudly
rather than publish another silently contracted dataset.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from psycopg import sql

from . import db


DEFAULT_SCHEMA = "padjective"
RECONCILIATION_TABLE = "taxonomy_path_reconciliation"

_NUMERIC_PATH_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_TAXONOMY_CODE_RE = re.compile(r"^(?P<root>[a-z]+)(?:-(?P<suffix>[0-9]+(?:-[0-9]+)*))?$")


class TaxonomyPathRepairError(RuntimeError):
    """Raised when taxonomy-path evidence is incomplete or contradictory."""


@dataclass(frozen=True)
class ReconciliationRow:
    """One current taxonomy and its recovered numeric hierarchy key."""

    taxonomy_id: str
    numeric_path: str
    observed_taxonomy_path: str | None
    resolution_source: str


@dataclass(frozen=True)
class ReconciliationReport:
    """Summary of a completed reconciliation pass."""

    taxonomy_count: int
    current_numeric_count: int
    historical_snapshot_count: int
    taxonomy_id_inference_count: int
    unresolved_taxonomy_ids: tuple[str, ...]

    @property
    def resolved_taxonomy_count(self) -> int:
        return (
            self.current_numeric_count
            + self.historical_snapshot_count
            + self.taxonomy_id_inference_count
        )


def is_numeric_taxonomy_path(path: str | None) -> bool:
    """Return whether ``path`` is a dotted numeric hierarchy key."""

    return bool(path and _NUMERIC_PATH_RE.fullmatch(path.strip()))


def _taxonomy_code(taxonomy_id: str) -> tuple[str, str | None] | None:
    code = taxonomy_id.rsplit("/", 1)[-1].lower()
    match = _TAXONOMY_CODE_RE.fullmatch(code)
    if match is None:
        return None
    return match.group("root"), match.group("suffix")


def build_reconciliation_rows(
    current_rows: Iterable[tuple[str, str | None]],
    historical_rows: Iterable[tuple[str, str | None]] = (),
    existing_rows: Iterable[tuple[str, str, str]] = (),
) -> tuple[list[ReconciliationRow], ReconciliationReport]:
    """Resolve numeric paths for current taxonomies from supplied evidence.

    ``existing_rows`` contains ``(taxonomy_id, numeric_path, source)`` tuples
    previously persisted by this module.  It makes subsequent reconciliation
    runs cheap while still checking any surviving live numeric path for drift.
    """

    current = {
        str(taxonomy_id): str(path) if path is not None else None
        for taxonomy_id, path in current_rows
    }
    historical = defaultdict(set)
    existing: dict[str, tuple[str, str]] = {}
    evidence = defaultdict(set)

    for taxonomy_id, path in current.items():
        if is_numeric_taxonomy_path(path):
            evidence[taxonomy_id].add(path.strip())

    for taxonomy_id, path in historical_rows:
        taxonomy_id = str(taxonomy_id)
        if is_numeric_taxonomy_path(path):
            numeric_path = str(path).strip()
            historical[taxonomy_id].add(numeric_path)
            evidence[taxonomy_id].add(numeric_path)

    for taxonomy_id, numeric_path, source in existing_rows:
        taxonomy_id = str(taxonomy_id)
        numeric_path = str(numeric_path).strip()
        if not is_numeric_taxonomy_path(numeric_path):
            raise TaxonomyPathRepairError(
                f"Persisted reconciliation for {taxonomy_id!r} is not numeric: "
                f"{numeric_path!r}"
            )
        existing[taxonomy_id] = (numeric_path, str(source))
        evidence[taxonomy_id].add(numeric_path)

    conflicts = {
        taxonomy_id: sorted(paths)
        for taxonomy_id, paths in evidence.items()
        if len(paths) > 1
    }
    if conflicts:
        taxonomy_id = sorted(conflicts)[0]
        raise TaxonomyPathRepairError(
            "Conflicting numeric paths for "
            f"{taxonomy_id!r}: {', '.join(conflicts[taxonomy_id])}"
        )

    known_paths = {
        taxonomy_id: next(iter(paths))
        for taxonomy_id, paths in evidence.items()
        if paths
    }

    root_numbers = defaultdict(set)
    for taxonomy_id, numeric_path in known_paths.items():
        taxonomy_code = _taxonomy_code(taxonomy_id)
        if taxonomy_code is None:
            continue
        root, _suffix = taxonomy_code
        root_numbers[root].add(numeric_path.split(".", 1)[0])

    root_conflicts = {
        root: sorted(numbers)
        for root, numbers in root_numbers.items()
        if len(numbers) > 1
    }
    if root_conflicts:
        root = sorted(root_conflicts)[0]
        raise TaxonomyPathRepairError(
            f"Conflicting numeric root mapping for {root!r}: "
            f"{', '.join(root_conflicts[root])}"
        )
    root_map = {root: next(iter(numbers)) for root, numbers in root_numbers.items()}

    rows: list[ReconciliationRow] = []
    unresolved: list[str] = []
    source_counts: Counter[str] = Counter()

    for taxonomy_id in sorted(current):
        observed_path = current[taxonomy_id]
        known_path = known_paths.get(taxonomy_id)
        taxonomy_code = _taxonomy_code(taxonomy_id)
        inferred_path = None
        if taxonomy_code is not None:
            root, suffix = taxonomy_code
            root_number = root_map.get(root)
            if root_number is not None:
                inferred_path = root_number
                if suffix:
                    inferred_path += "." + suffix.replace("-", ".")

        if known_path is not None and inferred_path is not None and known_path != inferred_path:
            raise TaxonomyPathRepairError(
                f"Taxonomy ID inference for {taxonomy_id!r} produced {inferred_path!r}, "
                f"but numeric evidence says {known_path!r}"
            )

        if is_numeric_taxonomy_path(observed_path):
            numeric_path = str(observed_path).strip()
            source = "current_numeric"
        elif taxonomy_id in existing:
            numeric_path, source = existing[taxonomy_id]
        elif historical.get(taxonomy_id):
            numeric_path = next(iter(historical[taxonomy_id]))
            source = "historical_snapshot"
        elif inferred_path is not None:
            numeric_path = inferred_path
            source = "taxonomy_id_inference"
        else:
            unresolved.append(taxonomy_id)
            continue

        source_counts[source] += 1
        rows.append(
            ReconciliationRow(
                taxonomy_id=taxonomy_id,
                numeric_path=numeric_path,
                observed_taxonomy_path=observed_path,
                resolution_source=source,
            )
        )

    report = ReconciliationReport(
        taxonomy_count=len(current),
        current_numeric_count=source_counts["current_numeric"],
        historical_snapshot_count=source_counts["historical_snapshot"],
        taxonomy_id_inference_count=source_counts["taxonomy_id_inference"],
        unresolved_taxonomy_ids=tuple(unresolved),
    )
    return rows, report


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return cur.fetchone() is not None


def _ensure_storage(conn, schema: str) -> None:
    db.ensure_schema(conn, schema)
    db.ensure_table(
        conn,
        schema,
        RECONCILIATION_TABLE,
        (
            "taxonomy_id TEXT PRIMARY KEY",
            "numeric_path TEXT NOT NULL",
            "observed_taxonomy_path TEXT",
            "resolution_source TEXT NOT NULL CHECK (resolution_source IN "
            "('current_numeric', 'historical_snapshot', 'taxonomy_id_inference'))",
            "resolved_at TIMESTAMPTZ NOT NULL DEFAULT now()",
            "last_verified_at TIMESTAMPTZ NOT NULL DEFAULT now()",
        ),
        indexes_sql=(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} ON {schema}.{table} "
                "(numeric_path) TABLESPACE pg_default"
            ).format(
                index=sql.Identifier(f"{schema}_taxonomy_path_reconciliation_numeric_idx"),
                schema=sql.Identifier(schema),
                table=sql.Identifier(RECONCILIATION_TABLE),
            ),
        ),
    )


def _fetch_current_rows(conn) -> list[tuple[str, str | None]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT taxonomy_id, taxonomy_path
            FROM cantbuymelove.taxonomy
            ORDER BY taxonomy_id
            """
        )
        return [
            (str(taxonomy_id), str(path) if path is not None else None)
            for taxonomy_id, path in cur.fetchall()
        ]


def _fetch_existing_rows(conn, schema: str) -> list[tuple[str, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT taxonomy_id, numeric_path, resolution_source "
                "FROM {schema}.{table} ORDER BY taxonomy_id"
            ).format(
                schema=sql.Identifier(schema),
                table=sql.Identifier(RECONCILIATION_TABLE),
            )
        )
        return [
            (str(taxonomy_id), str(path), str(source))
            for taxonomy_id, path, source in cur.fetchall()
        ]


def _fetch_historical_rows(conn, schema: str) -> list[tuple[str, str]]:
    table = "product_taxonomy_bench_products"
    if not _table_exists(conn, schema, table):
        return []
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT DISTINCT taxonomy_id, taxonomy_path "
                "FROM {schema}.{table}"
            ).format(schema=sql.Identifier(schema), table=sql.Identifier(table))
        )
        return [(str(taxonomy_id), str(path)) for taxonomy_id, path in cur.fetchall()]


def reconcile_taxonomy_paths(
    conn,
    *,
    schema: str = DEFAULT_SCHEMA,
    require_complete: bool = True,
) -> ReconciliationReport:
    """Populate and verify Padjective's numeric taxonomy-path map.

    Historical snapshots are scanned only when the reconciliation table does
    not yet cover every current taxonomy.  Normal pipeline calls therefore do
    a small live-table verification rather than repeatedly scanning snapshots.
    """

    _ensure_storage(conn, schema)
    current_rows = _fetch_current_rows(conn)
    existing_rows = _fetch_existing_rows(conn, schema)
    current_ids = {taxonomy_id for taxonomy_id, _path in current_rows}
    existing_ids = {taxonomy_id for taxonomy_id, _path, _source in existing_rows}
    historical_rows: Sequence[tuple[str, str | None]] = ()
    if not current_ids.issubset(existing_ids):
        historical_rows = _fetch_historical_rows(conn, schema)

    rows, report = build_reconciliation_rows(
        current_rows,
        historical_rows,
        existing_rows,
    )
    if require_complete and report.unresolved_taxonomy_ids:
        sample = ", ".join(report.unresolved_taxonomy_ids[:5])
        raise TaxonomyPathRepairError(
            f"Could not recover numeric paths for {len(report.unresolved_taxonomy_ids)} "
            f"current taxonomies (first: {sample})"
        )

    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.{table} (
                    taxonomy_id,
                    numeric_path,
                    observed_taxonomy_path,
                    resolution_source
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (taxonomy_id) DO UPDATE SET
                    numeric_path = EXCLUDED.numeric_path,
                    observed_taxonomy_path = EXCLUDED.observed_taxonomy_path,
                    resolution_source = EXCLUDED.resolution_source,
                    last_verified_at = now()
                """
            ).format(
                schema=sql.Identifier(schema),
                table=sql.Identifier(RECONCILIATION_TABLE),
            ),
            [
                (
                    row.taxonomy_id,
                    row.numeric_path,
                    row.observed_taxonomy_path,
                    row.resolution_source,
                )
                for row in rows
            ],
        )
    conn.commit()
    return report


def taxonomy_path_sql(*, table_alias: str = "tpr") -> sql.SQL:
    """Return the resolved taxonomy-path SQL expression for query builders."""

    return sql.SQL("COALESCE({alias}.numeric_path, t.taxonomy_path)").format(
        alias=sql.Identifier(table_alias)
    )


def reconciliation_join_sql(
    *, schema: str = DEFAULT_SCHEMA, table_alias: str = "tpr"
) -> sql.SQL:
    """Return a LEFT JOIN against the persisted reconciliation table."""

    return sql.SQL(
        "LEFT JOIN {schema}.{table} {alias} ON {alias}.taxonomy_id = t.taxonomy_id"
    ).format(
        schema=sql.Identifier(schema),
        table=sql.Identifier(RECONCILIATION_TABLE),
        alias=sql.Identifier(table_alias),
    )
