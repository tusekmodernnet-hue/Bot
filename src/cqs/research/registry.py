"""Append-only experiment registry (JSON Lines, committed to the repository).

Every trial is logged, including the ones that fail: the number of trials per hypothesis family is what the
Deflated Sharpe Ratio needs, and hiding failed trials would inflate any surviving result.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import uuid
from typing import Any

import numpy as np

from cqs import config


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=config.REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _code_version() -> dict:
    # the registry file itself is excluded: appending to it must not mark the code as modified
    dirty = _git("status", "--porcelain", "--", ".", ":(exclude)research/registry.jsonl")
    return {"commit": _git("rev-parse", "HEAD") or "uncommitted", "dirty": bool(dirty)}


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _append(record: dict) -> dict:
    record = _clean(record)
    config.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def log_experiment(family: str, name: str, params: dict, metrics: dict, data_fingerprint: str,
                   period: tuple[str, str], notes: str = "", sharpe_per_period: float | None = None) -> dict:
    """Log one trial. `family` is the pre-registered hypothesis family (H1..H7, BENCH, ...)."""
    return _append({
        "type": "experiment",
        "id": uuid.uuid4().hex[:12],
        "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "family": family,
        "name": name,
        "params": params,
        "metrics": metrics,
        "sharpe_per_period": sharpe_per_period,
        "data": data_fingerprint,
        "period": list(period),
        "code": _code_version(),
        "notes": notes,
    })


def log_event(event: str, payload: dict) -> dict:
    return _append({
        "type": "event",
        "event": event,
        "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code": _code_version(),
        **payload,
    })


def read() -> list[dict]:
    if not config.REGISTRY_PATH.exists():
        return []
    with open(config.REGISTRY_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def trials(family: str) -> list[dict]:
    return [r for r in read() if r.get("type") == "experiment" and r.get("family") == family]


def trial_count(family: str) -> int:
    return len(trials(family))


def sharpe_variance(family: str) -> float:
    """Variance of per-period Sharpe ratios across the family's trials (input to the Deflated Sharpe)."""
    srs = [t["sharpe_per_period"] for t in trials(family) if t.get("sharpe_per_period") is not None]
    return float(np.var(srs, ddof=1)) if len(srs) > 1 else float("nan")
