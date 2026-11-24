import io
import zipfile

import pandas as pd

from cqs import config
from cqs.data import parsers


def _zip(tmp_path, name, text):
    p = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(f"{name}.csv", text)
    return p


ROW_MS = "1577836800000,7189.43,7190.52,7170.15,7171.55,2449.049,1577840399999,17576274.30,3930,1323.785,9500855.15,0\n"
ROW_US = "1735689600000000,93576.00,94509.42,93489.03,94401.14,12.6,1735693199999999,1179000.0,2000,7.0,655000.0,0\n"


def test_klines_without_header(tmp_path):
    df = parsers.parse_klines(_zip(tmp_path, "a", ROW_MS))
    assert len(df) == 1
    assert df.open_time.iloc[0] == pd.Timestamp("2020-01-01 00:00", tz="UTC")
    assert df.close.iloc[0] == 7171.55


def test_klines_with_header(tmp_path):
    df = parsers.parse_klines(_zip(tmp_path, "b", ",".join(config.KLINE_COLUMNS) + "\n" + ROW_MS))
    assert len(df) == 1 and df.trades.iloc[0] == 3930


def test_microsecond_timestamps_detected(tmp_path):
    df = parsers.parse_klines(_zip(tmp_path, "c", ROW_US))
    assert df.open_time.iloc[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC")


def test_funding_time_jitter_rounded(tmp_path):
    text = "calc_time,funding_interval_hours,last_funding_rate\n1577865600007,8,-0.00012383\n"
    df = parsers.parse_funding(_zip(tmp_path, "f", text))
    assert df.funding_time.iloc[0] == pd.Timestamp("2020-01-01 08:00", tz="UTC")
    assert df.rate.iloc[0] == -0.00012383


def test_metrics_available_time_is_delayed(tmp_path):
    text = ",".join(config.METRICS_COLUMNS) + "\n2021-06-01 00:00:00,BTCUSDT,37821.5,1408551452.3,1.42,1.13,1.40,0.94\n"
    df = parsers.parse_metrics(_zip(tmp_path, "m", text))
    assert df.available_time.iloc[0] - df.create_time.iloc[0] == config.METRICS_DELAY
