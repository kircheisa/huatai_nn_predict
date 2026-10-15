import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "43_1_华泰证券债券预测_数据集预处理.py"
SPEC = importlib.util.spec_from_file_location("preprocess_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rolling_robust_standardize_short_window():
    arr = np.array([1.0, 2.0, 3.0])
    result = MODULE.rolling_robust_standardize(arr, 5)
    assert np.isnan(result).all()


def test_apply_rolling_robust_std_to_df_coerces_non_numeric():
    df = pd.DataFrame(
        {"Unnamed: 0": pd.date_range("2020-01-01", periods=3), "a": ["1", "x", "3"]}
    )
    result = MODULE.apply_rolling_robust_std_to_df(df, 2)
    assert result.shape == (3, 1)
    assert result.index.is_monotonic_increasing


def test_validate_required_columns_raises():
    df = pd.DataFrame({"x": [1]})
    try:
        MODULE.validate_required_columns(df, ["y"], "test")
    except ValueError as exc:
        assert "缺少列" in str(exc)
    else:
        assert False
