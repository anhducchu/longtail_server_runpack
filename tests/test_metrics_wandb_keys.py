import json
from pathlib import Path

from runner.metrics import collect_metrics


def test_shared_metrics_json_unwraps_to_clean_keys(tmp_path):
    """The shared evaluator wraps FID/IS/KID/PRD in a "metrics" key alongside
    "protocol"/"label_histogram" metadata. A naive full-payload flatten buried
    every metric under generation/metrics/FID and swept protocol constants in
    as if they were metrics; this must unwrap to generation/FID instead."""
    (tmp_path / "metrics.unified.json").write_text(json.dumps({
        "metrics": {"FID": 12.3, "IS": 8.1, "F_8": 0.6, "F_1_8": 0.4,
                    "ImprovedPrecision": 0.7, "Recall": 0.5, "KID": 0.01},
        "protocol": {"samples": 50000, "labels": "uniform support across classes"},
        "label_histogram": [1, 2, 3],
    }))
    out = collect_metrics(tmp_path)
    assert out["generation/FID"] == 12.3
    assert out["generation/KID"] == 0.01
    assert "generation/metrics/FID" not in out
    assert not any("protocol" in k or "label_histogram" in k for k in out)


def test_native_metric_file_is_collected(tmp_path):
    (tmp_path / "metrics.native.json").write_text(json.dumps({
        "metrics": {"FID": 12.3},
    }))
    assert collect_metrics(tmp_path)["generation/FID"] == 12.3


def test_per_class_metrics_keeps_only_the_tail_summary(tmp_path):
    """metrics.per_class.json carries a 100-entry per_class breakdown plus a
    Many/Medium/Few groups summary. Only the groups summary is useful in a
    W&B run's summary view; the 100-class detail belongs in the report CSV,
    not flooding every run with ~300 extra summary fields."""
    (tmp_path / "metrics.per_class.json").write_text(json.dumps({
        "protocol": {"reference": "balanced CIFAR train"},
        "per_class": {str(i): {"FID": i * 0.1, "generated": 500, "reference": 500} for i in range(100)},
        "groups": {
            "Many": {"FID": 10.1, "classes": [0, 1, 2], "generated": 15000, "reference": 15000},
            "Medium": {"FID": 15.2, "classes": [3, 4, 5], "generated": 15000, "reference": 15000},
            "Few": {"FID": 25.3, "classes": [6, 7, 8], "generated": 20000, "reference": 20000},
        },
    }))
    out = collect_metrics(tmp_path)
    assert out["generation/tail/Many/FID"] == 10.1
    assert out["generation/tail/Few/FID"] == 25.3
    assert not any("per_class" in k for k in out)
    assert len(out) < 15


def test_noncanonical_flat_metrics_json_is_ignored(tmp_path):
    """Only the shared evaluator's wrapped JSON contract is publishable."""
    (tmp_path / "metrics.extra.json").write_text(json.dumps({
        "protocol": "additional baseline metric output",
        "FID": 9.5, "KID": {"mean": 0.02, "std": 0.001, "all": [0.02, 0.021]},
        "num_generated": 20000, "num_reference": 20000, "seed": 0,
    }))
    assert collect_metrics(tmp_path) == {}
