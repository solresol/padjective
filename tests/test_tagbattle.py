from padjective.tagbattle import (
    filter_nested_tags,
    split_title,
    tag_positions,
    build_battles,
)


def test_filter_nested_tags():
    assert filter_nested_tags(["chocolate", "milk chocolate"]) == ["milk chocolate"]
    assert filter_nested_tags(["bunny", "big bunny", "bunny"]) == ["big bunny"]


def test_split_title():
    assert split_title("Easter bunny - milk chocolate") == ["Easter bunny", "milk chocolate"]
    assert split_title("SingleTitle") == ["SingleTitle"]


def test_tag_positions():
    pos = tag_positions("Nice red shoe", ["red", "shoe", "blue"])
    assert pos == {"red": 5, "shoe": 9}


def test_build_battles_generates_pairs():
    pairs = build_battles("big bunny milk chocolate", "bunny,milk chocolate")
    assert pairs == [("BUNNY", "MILK CHOCOLATE")]
