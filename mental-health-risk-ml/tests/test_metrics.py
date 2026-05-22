from src.evaluate import compute_metrics

def test_compute_metrics_basic():
    m = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.8, 0.4, 0.2])
    assert 0 <= m["accuracy"] <= 1
    assert "specificity" in m
