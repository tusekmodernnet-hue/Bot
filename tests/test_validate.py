import numpy as np
import pandas as pd

from cqs.data.validate import check_bars, strip_zombie_tail, zombie_tail_length


def bars(closes, vols, start="2022-01-01", freq="1h"):
    c = np.asarray(closes, float)
    return pd.DataFrame({
        "open_time": pd.date_range(start, periods=len(c), freq=freq, tz="UTC"),
        "open": c, "high": c, "low": c, "close": c, "volume": np.asarray(vols, float),
    })


def test_zombie_tail_detected_and_stripped():
    b = bars([1.0, 1.1, 1.2, 1.59, 1.59, 1.59], [5, 5, 5, 0, 0, 0])
    b.loc[2, "high"] = 1.3  # a real bar with range
    assert zombie_tail_length(b) == 3
    assert len(strip_zombie_tail(b)) == 3


def test_live_contract_has_no_zombie_tail():
    b = bars([1.0, 1.1, 1.2], [5, 5, 5])
    assert zombie_tail_length(b) == 0


def test_quiet_bar_with_volume_is_not_zombie():
    b = bars([1.0, 1.0], [5, 1])
    assert zombie_tail_length(b) == 0


def test_gaps_duplicates_and_bad_ohlc():
    b = bars([1, 2, 3, 4, 5], [1] * 5)
    b = pd.concat([b.drop(index=[2, 3]), b.iloc[[4]]])  # 2-bar gap + 1 duplicate
    b.loc[0, "high"] = 0.5  # high below open/close
    r = check_bars(b, "X", "1h")
    assert r.missing_bars == 2 and r.missing_blocks[0][1] == 2
    assert r.duplicate_bars == 1
    assert r.ohlc_inconsistent == 1
