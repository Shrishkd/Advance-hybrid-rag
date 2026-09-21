"""Experiment result store — append-only history in SQLite.

WHY A DATABASE WHEN WE ALREADY WRITE MARKDOWN REPORTS
------------------------------------------------------
The reports in `reports/` are for a human reading an argument. They are
rewritten every run, so they hold the CURRENT answer and no memory of how it
got there. That is the right shape for a report and the wrong shape for
catching a regression.

The question a report cannot answer is: *"recall@10 was 0.8229 last Tuesday
and it is 0.79 today - what changed?"* Answering that needs every run kept,
keyed by what produced it.

APPEND-ONLY, DELIBERATELY
-------------------------
There is no UPDATE and no DELETE in this module. A run that embarrasses us is
exactly the run worth keeping: the negative results are the evidence that the
positive ones were earned. Never overwrite history; regressions must stay
visible.

WHAT A ROW IS KEYED BY
----------------------
    config_hash   fingerprint of the config values that affect the result
    git_sha       code version, when the repo is under git
    dataset_ver   content hash of the golden set

All three are needed. A score is only reproducible if you know the config, the
code AND the questions - and the golden set is the one people forget. Add five
questions and every metric shifts for reasons that have nothing to do with
retrieval. Storing its hash makes that visible instead of mysterious.

git_sha is only READ, with a plain subprocess call, and it
degrades to "nogit" when the repo is not initialised. Nothing here ever
writes to git.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import time
from pathlib import Path

DB = Path("evaluation/results.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    experiment   TEXT    NOT NULL,   -- 'retrieval' | 'embedder' | ...
    config_name  TEXT    NOT NULL,   -- the swept value, e.g. 'hybrid_rrf'
    config_hash  TEXT    NOT NULL,
    git_sha      TEXT    NOT NULL,
    dataset_ver  TEXT    NOT NULL,
    n_scored     INTEGER NOT NULL,
    metrics      TEXT    NOT NULL,   -- JSON blob
    pinned       TEXT    NOT NULL    -- JSON of the pinned axes
);
CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(experiment, config_name, ts);
"""


def _connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA)
    return conn


def git_sha() -> str:
    """Current commit, or 'nogit'. Read-only — never initialises a repo.

    Returns 'nogit' when the directory is not a repository, and 'dirty-<sha>'
    when the working tree has uncommitted changes, because a score produced
    from uncommitted code is not reproducible from the sha alone.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode != 0:
            return "nogit"
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        prefix = "dirty-" if status.stdout.strip() else ""
        return prefix + sha.stdout.strip()
    except Exception:                                    # noqa: BLE001
        return "nogit"


def config_hash(cfg: dict) -> str:
    """Stable fingerprint of a config dict.

    `sort_keys` matters: without it, two identical configs whose keys happened
    to be inserted in a different order would hash differently and look like
    two separate experiments.

    >>> config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    True
    >>> config_hash({"a": 1}) == config_hash({"a": 2})
    False
    """
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def dataset_version(path: Path) -> str:
    """Content hash of the golden set.

    Content, not mtime: re-saving the same file unchanged must not look like a
    new dataset, and editing one answer must.
    """
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def record(
    experiment: str,
    config_name: str,
    metrics: dict,
    *,
    cfg: dict,
    pinned: dict,
    n_scored: int,
    dataset: Path,
) -> int:
    """Append one result row. Returns its id."""
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO runs (ts, experiment, config_name, config_hash, git_sha,"
        " dataset_ver, n_scored, metrics, pinned)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (time.time(), experiment, config_name, config_hash(cfg), git_sha(),
         dataset_version(dataset), n_scored,
         json.dumps(metrics, sort_keys=True), json.dumps(pinned, sort_keys=True)),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def history(experiment: str, config_name: str | None = None,
            limit: int = 20) -> list[dict]:
    """Past runs, newest first."""
    conn = _connect()
    q = "SELECT ts, config_name, config_hash, git_sha, dataset_ver, n_scored," \
        " metrics FROM runs WHERE experiment = ?"
    args: list = [experiment]
    if config_name:
        q += " AND config_name = ?"
        args.append(config_name)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    rows = [
        {"ts": r[0], "config": r[1], "config_hash": r[2], "git_sha": r[3],
         "dataset_ver": r[4], "n_scored": r[5], **json.loads(r[6])}
        for r in conn.execute(q, args)
    ]
    conn.close()
    return rows


def compare(experiment: str, config_name: str, metric: str = "recall@10",
            threshold: float = 0.02) -> tuple[bool, str]:
    """Compare the two most recent runs of one config.

    Returns (regressed, message). `threshold` is an ABSOLUTE drop, not
    relative: a 2-point fall on a 0.82 baseline and on a 0.20 baseline are
    equally worth knowing about, and relative thresholds make small numbers
    unfalsifiable.

    A 2-point default sits well inside the n=48 noise band, so this is a
    tripwire for investigation, not a verdict.
    """
    rows = history(experiment, config_name, limit=2)
    if len(rows) < 2:
        return False, f"only {len(rows)} run(s) for {config_name} — nothing to compare"
    now, prev = rows[0], rows[1]
    if metric not in now or metric not in prev:
        return False, f"metric {metric!r} missing from one of the runs"
    delta = now[metric] - prev[metric]
    note = ""
    if now["dataset_ver"] != prev["dataset_ver"]:
        # Without this the tool would blame the code for a dataset edit.
        note = ("  NOTE: golden set changed between runs "
                f"({prev['dataset_ver']} -> {now['dataset_ver']}); "
                "the delta is not attributable to code alone.")
    msg = (f"{config_name} {metric}: {prev[metric]:.4f} -> {now[metric]:.4f} "
           f"({delta:+.4f})" + note)
    return delta < -threshold, msg
