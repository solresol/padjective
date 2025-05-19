from padjective.ranking import compute_rankings


def test_compute_rankings_disjoint_components():
    pairs = [("A", "B"), ("B", "C"), ("X", "Y")]
    df = compute_rankings(pairs)
    # Expect two components
    assert sorted(df["component"].unique()) == [0, 1]
    comp1_tags = set(df[df["component"] == 1]["tag"])
    assert comp1_tags == {"X", "Y"}
    assert df.shape[0] == 5

