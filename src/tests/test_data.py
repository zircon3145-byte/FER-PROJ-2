import os
from src.data import preprocess


def test_data_folders_exist():
    assert os.path.exists("data/raw/train")
    assert os.path.exists("data/raw/test")
    assert os.path.exists("data/processed/train")
    assert os.path.exists("data/processed/validation")


def test_preprocess_runs():
    result = preprocess.preprocess_data()

    assert result is not None
