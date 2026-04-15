from padjective.product_taxonomy_bench import (
    SourceProductRow,
    build_tag_id_map,
    canonicalize_product_url,
    create_snapshot,
    first_title_occurrences,
    hash_product_url,
    parse_as_of,
    title_part_positions,
    title_parts,
)


def test_build_tag_id_map_is_deterministic() -> None:
    mapping = build_tag_id_map({"BETA", "ALPHA"}, width=4)
    assert mapping == {"ALPHA": "tag0001", "BETA": "tag0002"}


def test_canonicalize_product_url_prefers_stored_url() -> None:
    url = canonicalize_product_url("https://Example.com/products/x/?ref=1#frag")
    assert url == "https://example.com/products/x"


def test_canonicalize_product_url_falls_back_to_myshopify_domain() -> None:
    url = canonicalize_product_url(
        None, myshopify_domain="Shop.MyShopify.com", product_handle="my-product"
    )
    assert url == "https://shop.myshopify.com/products/my-product"


def test_hash_product_url_is_stable() -> None:
    digest = hash_product_url("https://example.com/products/x")
    assert digest == "7ba98a76d65fad5dc0cb09de73e83880f912fe7b21dad263b55793a201aee1d7"


def test_title_part_positions_and_first_occurrence() -> None:
    title = "Walking boots - men's leather"
    tags = ["WALKING", "BOOTS", "MEN'S", "LEATHER", "ABSENT"]

    parts = title_parts(title)
    assert parts == ["Walking boots", "men's leather"]

    positions = title_part_positions(title, tags)
    assert positions[0] == {"WALKING": 0, "BOOTS": 8}
    assert positions[1] == {"MEN'S": 0, "LEATHER": 6}

    first = first_title_occurrences(positions)
    assert first["WALKING"] == (0, 0)
    assert first["BOOTS"] == (0, 8)
    assert first["MEN'S"] == (1, 0)
    assert first["LEATHER"] == (1, 6)
    assert "ABSENT" not in first


def test_parse_as_of_accepts_iso8601_z() -> None:
    dt = parse_as_of("2026-02-11T19:15:00Z")
    assert dt.isoformat() == "2026-02-11T19:15:00+00:00"


def test_parse_as_of_accepts_utc_suffix_format() -> None:
    dt = parse_as_of("2026-02-11 19:15 UTC")
    assert dt.isoformat() == "2026-02-11T19:15:00+00:00"


class _SnapshotCursor:
    def __init__(self, connection: "_SnapshotConnection") -> None:
        self.connection = connection
        self._fetchone = None

    def __enter__(self) -> "_SnapshotCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query, params=None) -> None:
        text = str(query)
        self.connection.statements.append((text, params))
        if "SELECT COUNT(*) FROM" in text and "product_taxonomy_bench_products" in text:
            self._fetchone = (len(self.connection.inserted_product_hashes),)
        else:
            self._fetchone = None
            if "UPDATE" in text and "product_taxonomy_bench_snapshots" in text:
                self.connection.snapshot_update_params = params

    def executemany(self, query, params_seq) -> None:
        rows = list(params_seq)
        text = str(query)
        self.connection.statements.append((text, rows))
        if "INSERT INTO" in text and "product_taxonomy_bench_products" in text:
            for row in rows:
                self.connection.inserted_product_hashes.add(row[1])

    def fetchone(self):
        return self._fetchone


class _SnapshotConnection:
    def __init__(self) -> None:
        self.statements = []
        self.inserted_product_hashes: set[str] = set()
        self.snapshot_update_params = None
        self.commits = 0
        self.closed = False

    def cursor(self, *_, **__) -> _SnapshotCursor:
        return _SnapshotCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


def test_create_snapshot_uses_persisted_product_count_after_hash_dedup(monkeypatch) -> None:
    conn = _SnapshotConnection()

    rows = [
        SourceProductRow(
            product_id=1,
            title="Red Shirt",
            product_url="https://example.com/products/red-shirt?variant=1",
            myshopify_domain="example.myshopify.com",
            product_handle="red-shirt",
            raw_tags="RED,SHIRT",
            taxonomy_id="tax-1",
            taxonomy_path="1.1",
            taxonomy_name="Apparel > Shirts",
        ),
        SourceProductRow(
            product_id=2,
            title="Red Shirt",
            product_url="https://example.com/products/red-shirt?variant=2",
            myshopify_domain="example.myshopify.com",
            product_handle="red-shirt",
            raw_tags="RED,SHIRT",
            taxonomy_id="tax-1",
            taxonomy_path="1.1",
            taxonomy_name="Apparel > Shirts",
        ),
    ]

    monkeypatch.setattr("padjective.product_taxonomy_bench.db.get_connection", lambda _dsn: conn)
    monkeypatch.setattr("padjective.product_taxonomy_bench._ensure_storage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("padjective.product_taxonomy_bench._snapshot_name_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("padjective.product_taxonomy_bench.calculate_cv_folds", lambda *_args, **_kwargs: {1: 0, 2: 1})
    monkeypatch.setattr("padjective.product_taxonomy_bench._count_taxonomies", lambda *_args, **_kwargs: {"tax-1": 2})
    monkeypatch.setattr("padjective.product_taxonomy_bench._count_tags", lambda *_args, **_kwargs: {"RED": 2, "SHIRT": 2})
    monkeypatch.setattr("padjective.product_taxonomy_bench._stream_source_products", lambda *_args, **_kwargs: iter(rows))
    monkeypatch.setattr("padjective.product_taxonomy_bench._git_head", lambda: "deadbeef")

    create_snapshot(
        dsn=None,
        schema="padjective",
        snapshot_name="paper-dup-test",
        min_tag_count=1,
        min_samples_per_taxonomy=1,
    )

    assert conn.snapshot_update_params is not None
    assert conn.snapshot_update_params[0] == 1
    assert len(conn.inserted_product_hashes) == 1
