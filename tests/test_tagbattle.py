import sqlite3

from padjective.tagbattle import (
    filter_nested_tags,
    split_title,
    tag_positions,
    process_product,
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


def test_process_product_inserts_pairs():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE battles (winner_tag TEXT, loser_tag TEXT)")
    process_product("big bunny milk chocolate", "bunny,milk chocolate", cur)
    cur.execute("SELECT winner_tag, loser_tag FROM battles")
    rows = cur.fetchall()
    assert rows == [("BUNNY", "MILK CHOCOLATE")]
    conn.close()
