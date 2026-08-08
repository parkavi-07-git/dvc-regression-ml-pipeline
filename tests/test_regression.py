
import os
import json


def test_dvc_file_exists():
    assert os.path.exists("dvc.yaml")


def test_params_file_exists():
    assert os.path.exists("params.yaml")


def test_model_exists():
    assert os.path.exists("model.pkl")


def test_metrics_exists():
    assert os.path.exists("metrics.json")


def test_metrics_valid():
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    assert isinstance(metrics, dict)
    assert len(metrics) > 0
