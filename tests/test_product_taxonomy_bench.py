from padjective.product_taxonomy_bench import (
    build_tag_id_map,
    canonicalize_product_url,
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
