import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "model_nn" / "05_基础特征对照训练.py"
SPEC = importlib.util.spec_from_file_location("raw_training", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_published_value_cannot_arrive_early():
    dates = pd.date_range("2020-01-01", periods=5)
    values = pd.Series([10.0, 999.0])
    result = MODULE.align_published(values, pd.to_datetime(["2020-01-01", "2020-01-04"]), dates)
    assert np.isnan(result[0])
    np.testing.assert_array_equal(result[1:4], [10.0] * 3)
    assert result[4] == 999.0


def test_future_values_do_not_change_scaler():
    raw = np.column_stack([np.arange(30, dtype=float), np.full(30, np.nan)])
    raw[2, 0] = np.nan
    scaled, keep, state = MODULE.standardize(raw, np.arange(20))
    # 第一列只有19个观测，显式加入一行使其达到筛列下限。
    scaled, keep, state = MODULE.standardize(raw, np.arange(21))
    changed = raw.copy()
    changed[21:, 0] = 1e9
    other, other_keep, other_state = MODULE.standardize(changed, np.arange(21))
    np.testing.assert_array_equal(keep, [True, False])
    np.testing.assert_array_equal(keep, other_keep)
    np.testing.assert_allclose(scaled[:21], other[:21])
    np.testing.assert_allclose(state["median"], other_state["median"])
    assert scaled[2, 1] == 1


def test_metrics_use_predictions_not_correct_flags():
    result = MODULE.metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.6]))
    assert result["acc"] == 0.5
    assert result["pred_down"] == 1.0
    assert result["majority_baseline"] == 0.5


def test_rolling_normalization_is_prefix_invariant():
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    before = MODULE.causal_rolling_features(frame)
    changed = frame.copy()
    changed.loc[70:, "x"] = 100000.0
    after = MODULE.causal_rolling_features(changed)
    np.testing.assert_allclose(before.iloc[:70], after.iloc[:70], equal_nan=True)


def test_compact_gru_output_shape():
    import torch
    model = MODULE.CompactGRU(12, 16, 1)
    assert model(torch.zeros(2, 20, 12)).shape == (2, 2)


def optimization_module():
    path = PATH.parent / "09_方向优化实验.py"
    spec = importlib.util.spec_from_file_location("direction_optimization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direction_and_material_down_labels_are_distinct():
    module = optimization_module()
    targets = module.make_targets(np.array([-0.03, -0.01, 0.0, 0.03]))
    np.testing.assert_array_equal(targets["down"], [1, 1, 0, 0])
    np.testing.assert_array_equal(targets["down_ex2"], [1, 0, 0, 0])
    np.testing.assert_array_equal(targets["big"], [1, 0, 0, 1])


def test_filtered_auc_uses_probability_and_reports_coverage():
    module = optimization_module()
    result = module.seg_metrics(
        np.array([0, 1, 0, 1]), np.array([1, 1, 1, 1]),
        np.array([0.6, 0.9, 0.7, 0.8]), np.array([True, True, False, False]))
    assert result["acc"] == 0.5
    assert result["auc"] == 1.0
    assert result["coverage"] == 0.5


def test_block_interval_is_repeatable_and_handles_short_samples():
    module = optimization_module()
    assert module.block_interval(np.ones(40), repeats=20) == [1.0, 1.0]
    assert module.block_interval(np.ones(10)) is None
    values = np.arange(80, dtype=float) % 2
    assert module.block_interval(values, repeats=20) == module.block_interval(values, repeats=20)


def event_module():
    return optimization_module().text2


def sample_events(dates, directions=None):
    n = len(dates)
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "category": ["货币宽松"] * n,
        "direction": directions if directions is not None else [1] * n,
        "intensity": [2] * n,
    })


def test_events_only_available_on_next_trading_day():
    module = event_module()
    days = pd.bdate_range("2025-10-17", periods=4)
    events = sample_events(["2025-10-17", "2025-10-19", "2025-10-20"])
    result = module.build_event_features(days, events)
    assert result.iloc[0]["txt_cnt20"] == 0
    assert result.iloc[1]["txt_cnt20"] == 2
    assert result.iloc[1]["txt_pos"] == 4
    assert result.iloc[2]["txt_cnt20"] == 3


def test_future_events_cannot_change_past_features():
    module = event_module()
    days = pd.bdate_range("2025-01-01", periods=40)
    before = module.build_event_features(days, sample_events([days[1]]))
    after = module.build_event_features(days, sample_events([days[1], days[25]]))
    pd.testing.assert_frame_equal(before.iloc[:26], after.iloc[:26])


def test_objective_features_do_not_depend_on_direction():
    module = event_module()
    days = pd.bdate_range("2025-01-01", periods=30)
    positive = sample_events([days[0]], [1])
    negative = sample_events([days[0]], [-1])
    first = module.build_event_features(days, positive, "objective")
    second = module.build_event_features(days, negative, "objective")
    pd.testing.assert_frame_equal(first, second)
    assert not set(module.DIRECTION_COLUMNS).intersection(first.columns)


def test_event_decay_counts_and_neutral_intensity():
    module = event_module()
    days = pd.bdate_range("2025-01-01", periods=65)
    result = module.build_event_features(days, sample_events([days[0]], [0]))
    assert result.iloc[1]["txt_intensity"] == 2
    assert result.iloc[11]["txt_intensity"] == 1
    assert result.iloc[20]["txt_cnt20"] == 1
    assert result.iloc[21]["txt_cnt20"] == 0
    assert result.iloc[60]["txt_gap"] == 59
    assert result.iloc[61]["txt_gap"] == 60
    assert result.iloc[61]["txt_intensity"] == 0
    assert (result["txt_pos"] == 0).all()


def test_event_reader_accepts_tab_calendar_and_rejects_invalid_direction(tmp_path):
    import pytest
    module = event_module()
    path = tmp_path / "events.csv"
    events = sample_events(["2025-01-01"])
    events.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")
    assert len(module.read_events(path)) == 1
    events["direction"] = 9
    events.to_csv(path, index=False)
    with pytest.raises(ValueError, match="direction"):
        module.read_events(path)


def test_event_calendar_rejects_duplicates_and_truncated_history():
    import pytest
    module = event_module()
    events = sample_events(["2025-01-02"])
    with pytest.raises(ValueError, match="不重复"):
        module.build_event_features(["2025-01-01", "2025-01-01"], events)
    with pytest.raises(ValueError, match="完整交易日历"):
        module.build_event_features(pd.bdate_range("2025-01-06", periods=3), events)


def test_events_beyond_calendar_do_not_get_backfilled():
    module = event_module()
    days = pd.bdate_range("2025-01-01", periods=5)
    result = module.build_event_features(days, sample_events(["2025-01-20"]))
    assert (result["txt_cnt20"] == 0).all()
    assert (result["txt_intensity"] == 0).all()
    assert (result["txt_gap"] == 60).all()
