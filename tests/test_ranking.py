from padjective.ranking import compute_rankings


def test_compute_rankings_disjoint_components():
    pairs = [("A", "B"), ("B", "C"), ("X", "Y")]
    df = compute_rankings(pairs)
    # Expect two components
    assert sorted(df["component"].unique()) == [0, 1]
    # Component 1 has single tag 'X' or 'Y', score 0
    assert df[df["tag"] == "X"]["score"].iloc[0] == 0.0 or df[df["tag"] == "Y"]["score"].iloc[0] == 0.0
    assert df.shape[0] == 5

