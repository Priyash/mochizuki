"""SQLite ledger: every job, what it cost, what the critic said, and who approved what.

All spending decisions read from here, so a crash or a closed laptop loses nothing.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  key TEXT PRIMARY KEY,
  shot_id TEXT NOT NULL,
  chapter INTEGER NOT NULL,
  kind TEXT NOT NULL,            -- keyframe | variant | take | upscale
  idx INTEGER NOT NULL,
  model TEXT NOT NULL,
  cost REAL NOT NULL,
  status TEXT NOT NULL,          -- pending | done | failed
  path TEXT,
  verdict TEXT,                  -- NULL | pass | fail
  verdict_by TEXT,
  note TEXT,
  selected INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
  chapter INTEGER NOT NULL,
  gate TEXT NOT NULL,            -- board | animatic
  by TEXT NOT NULL,
  at REAL NOT NULL,
  PRIMARY KEY (chapter, gate)
);
CREATE TABLE IF NOT EXISTS batches (
  id TEXT PRIMARY KEY,
  stage TEXT NOT NULL,           -- keyframes | video | upscale
  chapter INTEGER NOT NULL,
  status TEXT NOT NULL,          -- built | submitted | done | failed
  est_seconds REAL NOT NULL,
  gpu_seconds REAL,              -- session time Kaggle charged, as the worker measured it
  created_at REAL NOT NULL,
  submitted_at REAL,
  finished_at REAL,
  note TEXT
);
"""

NOT_A_FAILURE = ("cut off", "stale inputs", "not reached")   # notes that do not count against a shot


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: str | Path, total_cap: float, chapter_caps: dict):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(jobs)")}
        for col, ddl in (("batch", "TEXT"), ("gpu_seconds", "REAL")):
            if col not in cols:
                self.db.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ddl}")
        self.db.commit()
        self.total_cap = float(total_cap)
        self.chapter_caps = {int(k): float(v) for k, v in chapter_caps.items()}

    # ---- spend. A pending job counts as spent: its outcome is unknown.
    def spent(self, chapter: int | None = None) -> float:
        q = "SELECT COALESCE(SUM(cost),0) FROM jobs WHERE 1=1"   # failed jobs carry cost 0 unless billed
        one = chapter is not None
        row = self.db.execute(q + (" AND chapter=?" if one else ""),
                              (chapter,) if one else ()).fetchone()
        return round(row[0], 4)

    def check_budget(self, chapter: int, cost: float) -> None:
        if cost <= 0:
            return
        if self.spent() + cost > self.total_cap + 1e-9:
            raise BudgetExceeded(
                f"total cap ${self.total_cap:.2f}: spent ${self.spent():.2f}, job ${cost:.2f}")
        cap = self.chapter_caps.get(chapter)
        if cap is not None and self.spent(chapter) + cost > cap + 1e-9:
            raise BudgetExceeded(
                f"chapter {chapter} cap ${cap:.2f}: spent ${self.spent(chapter):.2f}, job ${cost:.2f}")

    # ---- jobs
    def get(self, key: str):
        return self.db.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()

    def jobs(self, shot_id: str, kind: str | None = None) -> list:
        q, a = "SELECT * FROM jobs WHERE shot_id=?", [shot_id]
        if kind:
            q, a = q + " AND kind=?", a + [kind]
        return self.db.execute(q + " ORDER BY kind, idx", a).fetchall()

    def failures(self, shot_id: str, kind: str) -> int:
        rows = [j for j in self.jobs(shot_id, kind) if j["status"] == "failed"]
        return sum(1 for j in rows if (j["note"] or "") not in NOT_A_FAILURE)

    def begin(self, key, shot_id, chapter, kind, idx, model, cost, batch: str | None = None) -> bool:
        """Reserve a job before the provider is called. False if it already exists."""
        if self.get(key):
            return False
        self.check_budget(chapter, cost)
        self.db.execute(
            "INSERT INTO jobs(key,shot_id,chapter,kind,idx,model,cost,status,created_at,batch)"
            " VALUES(?,?,?,?,?,?,?, 'pending', ?, ?)",
            (key, shot_id, chapter, kind, idx, model, cost, time.time(), batch))
        self.db.commit()
        return True

    def finish(self, key: str, path: str | None, ok: bool = True,
               gpu_seconds: float | None = None, note: str | None = None) -> None:
        # A failed call is recorded at zero cost. Check the provider's dashboard if in doubt.
        if ok:
            self.db.execute("UPDATE jobs SET status='done', path=?, gpu_seconds=?, note=COALESCE(?, note)"
                            " WHERE key=?", (path, gpu_seconds, note, key))
        else:
            self.db.execute("UPDATE jobs SET status='failed', cost=0, gpu_seconds=?, note=? WHERE key=?",
                            (gpu_seconds, note, key))
        self.db.commit()

    def resolve(self, key: str, charged: bool) -> None:
        """Close a job whose call was cut off. Keep its cost if the provider billed it."""
        job = self.get(key)
        if not job or job["status"] != "pending":
            raise ValueError(f"{key} is not a pending job")
        self.db.execute("UPDATE jobs SET status='failed', cost=?, note=? WHERE key=?",
                        (job["cost"] if charged else 0, "cut off", key))
        self.db.commit()

    def pending(self) -> list:
        return self.db.execute("SELECT * FROM jobs WHERE status='pending'").fetchall()

    # ---- review
    def verdict(self, key: str, verdict: str, by: str, note: str = "") -> None:
        if verdict not in ("pass", "fail"):
            raise ValueError("verdict must be pass or fail")
        job = self.get(key)
        if not job:
            raise KeyError(key)
        if job["status"] != "done":
            raise ValueError(f"{key} has no finished output to judge")
        self.db.execute("UPDATE jobs SET verdict=?, verdict_by=?, note=? WHERE key=?",
                        (verdict, by, note, key))
        self.db.commit()

    def select(self, key: str) -> None:
        job = self.get(key)
        if not job or job["kind"] not in ("keyframe", "ref") or job["status"] != "done":
            raise ValueError(f"{key} is not a finished keyframe or reference still")
        self.db.execute("UPDATE jobs SET selected=0 WHERE shot_id=? AND kind=?",
                        (job["shot_id"], job["kind"]))
        self.db.execute("UPDATE jobs SET selected=1, verdict=COALESCE(verdict,'pass') WHERE key=?",
                        (key,))
        self.db.commit()

    def unselect(self, key: str, note: str = "") -> None:
        """A person rejects a selection: it becomes a failed candidate."""
        self.db.execute("UPDATE jobs SET selected=0, verdict='fail', verdict_by='human', note=? WHERE key=?",
                        (note, key))
        self.db.commit()

    def selected(self, shot_id: str):
        return self.db.execute(
            "SELECT * FROM jobs WHERE shot_id=? AND kind IN ('keyframe','ref') AND selected=1",
            (shot_id,)).fetchone()

    # ---- gates
    def approve(self, chapter: int, gate: str, by: str) -> None:
        if by != "human":
            raise PermissionError("gates are approved by a person, not by an agent")
        self.db.execute("INSERT OR REPLACE INTO approvals VALUES(?,?,?,?)",
                        (chapter, gate, by, time.time()))
        self.db.commit()

    def approved(self, chapter: int, gate: str) -> bool:
        return self.db.execute("SELECT 1 FROM approvals WHERE chapter=? AND gate=?",
                               (chapter, gate)).fetchone() is not None

    # ---- Kaggle batches
    def create_batch(self, batch_id: str, stage: str, chapter: int, est_seconds: float) -> None:
        self.db.execute("INSERT INTO batches(id,stage,chapter,status,est_seconds,created_at)"
                        " VALUES(?,?,?,'built',?,?)", (batch_id, stage, chapter, est_seconds, time.time()))
        self.db.commit()

    def batch(self, batch_id: str):
        return self.db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()

    def set_batch(self, batch_id: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE batches SET {cols} WHERE id=?", (*fields.values(), batch_id))
        self.db.commit()

    def active_batch(self):
        return self.db.execute("SELECT * FROM batches WHERE status IN ('built','submitted')"
                               " ORDER BY created_at LIMIT 1").fetchone()

    def batch_jobs(self, batch_id: str) -> list:
        return self.db.execute("SELECT * FROM jobs WHERE batch=? ORDER BY shot_id, kind, idx",
                               (batch_id,)).fetchall()

    def gpu_seconds_since(self, since: float) -> float:
        """Kaggle session time used by batches started since a moment: measured if known, else estimated."""
        row = self.db.execute(
            "SELECT COALESCE(SUM(COALESCE(gpu_seconds, est_seconds)),0) FROM batches"
            " WHERE status IN ('submitted','done','failed') AND COALESCE(submitted_at, created_at) >= ?",
            (since,)).fetchone()
        return float(row[0])
