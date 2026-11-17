"""
registry.py
===========
File-based model registry: the single source of truth for what the API serves.

Layout::

    models/
      registry.json              {"approved": <id|null>, "history": [...], "events": [...]}
      versions/<version_id>/     model.npz, metadata.json, movies.csv (optional)

Rules
-----
* Registering a version never changes what is served.
* Only :meth:`ModelRegistry.approve` changes the approved pointer, and every call
  records *why* (the decision record) in ``events``.
* Earlier approved versions stay on disk, so :meth:`rollback` is always possible.
* ``registry.json`` is written atomically (temp file + ``os.replace``) under a
  lock file, so a crash can never leave a half-written pointer.

MLflow is used for experiment tracking; this registry decides serving. Keeping the
serving decision in a plain JSON file means the API needs no MLflow server.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_FILE = "registry.json"
VERSIONS_DIR = "versions"


class RegistryError(RuntimeError):
    """Raised for invalid registry operations (unknown version, no approved model...)."""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ModelRegistry:
    """Versioned model storage with an explicit, auditable approval pointer."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / VERSIONS_DIR
        self.registry_path = self.root / REGISTRY_FILE
        self.lock_path = self.root / ".registry.lock"

    # ── internal helpers ──────────────────────────────────────────────────────
    @contextmanager
    def _lock(self, timeout: float = 30.0) -> Iterator[None]:
        """Cross-platform exclusive lock using O_CREAT | O_EXCL on a lock file."""
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                # Stale lock older than the timeout (crashed writer) is removed.
                try:
                    if time.time() - self.lock_path.stat().st_mtime > timeout:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise RegistryError(f"Timed out waiting for registry lock {self.lock_path}") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)

    def _read(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"approved": None, "history": [], "events": []}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.registry_path)

    # ── queries ───────────────────────────────────────────────────────────────
    def state(self) -> dict[str, Any]:
        return self._read()

    def version_dir(self, version_id: str) -> Path:
        if not version_id or "/" in version_id or "\\" in version_id or version_id.startswith("."):
            raise RegistryError(f"Invalid version id: {version_id!r}")
        return self.versions_dir / version_id

    def exists(self, version_id: str) -> bool:
        return (self.version_dir(version_id) / "metadata.json").exists()

    def metadata(self, version_id: str) -> dict[str, Any]:
        path = self.version_dir(version_id) / "metadata.json"
        if not path.exists():
            raise RegistryError(f"Unknown model version: {version_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_versions(self) -> list[str]:
        if not self.versions_dir.exists():
            return []
        return sorted(p.name for p in self.versions_dir.iterdir()
                      if (p / "metadata.json").exists())

    def approved_version(self) -> str | None:
        return self._read().get("approved")

    def approved_dir(self) -> Path:
        v = self.approved_version()
        if not v:
            raise RegistryError(f"No approved model in registry {self.root}")
        d = self.version_dir(v)
        if not (d / "metadata.json").exists():
            raise RegistryError(f"Approved version {v} is missing from {self.versions_dir}")
        return d

    # ── mutations ─────────────────────────────────────────────────────────────
    def register(self, source_dir: str | Path, version_id: str, metadata: dict[str, Any]) -> Path:
        """Copy a built model directory into the registry as a new version.

        Idempotent: registering an existing version id with identical metadata
        content hash is a no-op; a different one raises.
        """
        src = Path(source_dir)
        dest = self.version_dir(version_id)
        with self._lock():
            if dest.exists():
                existing = self.metadata(version_id)
                if existing.get("content_sha256") and existing.get("content_sha256") == metadata.get("content_sha256"):
                    logger.info("Version %s already registered (identical content); reusing", version_id)
                    return dest
                raise RegistryError(f"Version {version_id} already exists with different content")
            tmp = dest.with_name(dest.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp)
            meta = dict(metadata)
            meta.setdefault("version_id", version_id)
            meta.setdefault("registered_at", utc_now())
            meta["status"] = "registered"
            (tmp / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
            os.replace(tmp, dest)
            state = self._read()
            state.setdefault("events", []).append(
                {"time": utc_now(), "event": "registered", "version": version_id})
            self._write(state)
        logger.info("Registered model version %s", version_id)
        return dest

    def approve(self, version_id: str, decision: dict[str, Any]) -> None:
        """Make ``version_id`` the served model. ``decision`` records the evidence."""
        if not self.exists(version_id):
            raise RegistryError(f"Cannot approve unknown version {version_id}")
        if not decision or "reason" not in decision:
            raise RegistryError("An approval needs a decision record with a 'reason'")
        with self._lock():
            state = self._read()
            previous = state.get("approved")
            if previous == version_id:
                logger.info("Version %s is already approved", version_id)
                return
            if previous:
                state.setdefault("history", []).append(previous)
            state["approved"] = version_id
            state.setdefault("events", []).append({
                "time": utc_now(), "event": "approved", "version": version_id,
                "previous": previous, "decision": decision,
            })
            self._write(state)
            self._set_status(version_id, "approved")
            if previous:
                self._set_status(previous, "superseded")
        logger.info("Approved model version %s (previous: %s)", version_id, previous)

    def record_rejection(self, version_id: str, decision: dict[str, Any]) -> None:
        """Record that a candidate failed its promotion gate (it stays on disk)."""
        with self._lock():
            state = self._read()
            state.setdefault("events", []).append({
                "time": utc_now(), "event": "rejected", "version": version_id,
                "decision": decision,
            })
            self._write(state)
            if self.exists(version_id):
                self._set_status(version_id, "rejected")

    def rollback(self, reason: str = "manual rollback") -> str:
        """Re-approve the most recent previously approved version."""
        with self._lock():
            state = self._read()
            history: list[str] = state.get("history", [])
            current = state.get("approved")
            while history:
                candidate = history.pop()
                if self.exists(candidate) and candidate != current:
                    state["approved"] = candidate
                    state.setdefault("events", []).append({
                        "time": utc_now(), "event": "rollback", "version": candidate,
                        "previous": current, "decision": {"reason": reason},
                    })
                    self._write(state)
                    self._set_status(candidate, "approved")
                    if current and self.exists(current):
                        self._set_status(current, "rolled_back")
                    logger.warning("Rolled back from %s to %s", current, candidate)
                    return candidate
            raise RegistryError("No previous approved version available for rollback")

    def _set_status(self, version_id: str, status: str) -> None:
        path = self.version_dir(version_id) / "metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["status"] = status
        meta.setdefault("status_history", []).append({"time": utc_now(), "status": status})
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
