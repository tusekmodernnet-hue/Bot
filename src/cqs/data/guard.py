"""Holdout lock.

Research code must not see data from HOLDOUT_START onwards. The holdout is opened once, for the final test of
candidates that passed walk-forward validation, and every opening is logged in the experiment registry.

Opening needs two keys, so it cannot happen by accident:
1. the environment variable CQS_ALLOW_HOLDOUT=1, and
2. the `final_test(...)` context manager with a written reason.
"""

from __future__ import annotations

import contextlib
import contextvars
import os

import pandas as pd

from cqs import config

_final_test_reason: contextvars.ContextVar[str | None] = contextvars.ContextVar("final_test_reason", default=None)


class HoldoutAccessError(PermissionError):
    pass


def holdout_open() -> bool:
    return os.environ.get("CQS_ALLOW_HOLDOUT") == "1" and _final_test_reason.get() is not None


@contextlib.contextmanager
def final_test(reason: str):
    """Open the holdout for the duration of the block. Logged in the registry."""
    if not reason or len(reason.strip()) < 10:
        raise ValueError("final_test needs a written reason")
    if os.environ.get("CQS_ALLOW_HOLDOUT") != "1":
        raise HoldoutAccessError("set CQS_ALLOW_HOLDOUT=1 to run a declared final test")
    from cqs.research.registry import log_event  # local import: registry depends on config only

    log_event("holdout_opened", {"reason": reason})
    token = _final_test_reason.set(reason)
    try:
        yield
    finally:
        _final_test_reason.reset(token)
        log_event("holdout_closed", {"reason": reason})


def resolve_end(end: pd.Timestamp | str | None) -> pd.Timestamp:
    """End of the allowed data range. None means 'as far as allowed'."""
    if end is None:
        return config.HOLDOUT_END if holdout_open() else config.HOLDOUT_START - pd.Timedelta(1, "us")
    end = pd.Timestamp(end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end >= config.HOLDOUT_START and not holdout_open():
        raise HoldoutAccessError(
            f"requested data up to {end}, but the holdout starts {config.HOLDOUT_START} and is locked"
        )
    return end


def enforce(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Drop holdout rows unless a final test is open. Last line of defence for any loader."""
    if holdout_open():
        return df
    return df[df[time_col] < config.HOLDOUT_START]
