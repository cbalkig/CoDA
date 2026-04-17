# render_logger.py
#
# Performance-friendly logger that streams rows to CSV while keeping the
# public API unchanged (record / has_all / has / close).

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class RenderLogger:
    # ------------------------------------------------------------------ #
    def __init__(
            self,
            success_csv: Path,
            failure_csv: Optional[Path] = None,
            *,
            chunk_size: int = 100,
            add_timestamp: bool = True,
    ) -> None:
        self.success_csv = Path(success_csv)
        self.failure_csv = Path(failure_csv) if failure_csv else None
        self.chunk_size = chunk_size
        self.add_timestamp = add_timestamp

        # ring buffers (stay small)
        self._succ_buffer: List[Dict[str, Any]] = []
        self._fail_buffer: List[Dict[str, Any]] = []

        # header flags
        self._succ_header_written = (
                self.success_csv.exists() and self.success_csv.stat().st_size > 0
        )
        self._fail_header_written = (
                self.failure_csv is not None
                and self.failure_csv.exists()
                and self.failure_csv.stat().st_size > 0
        )

        # success tracking
        self._success_counts: dict[str, int] = defaultdict(int)  # md5 -> #images
        self._success_paths: dict[str, set[str]] = defaultdict(set)  # md5 -> {img paths}
        self._all_success_paths: set[str] = set()  # flat set for O(1) has()

        # seed from existing CSV (if any)
        self._load_success_counts()

        # make sure folders exist
        self.success_csv.parent.mkdir(parents=True, exist_ok=True)
        if self.failure_csv:
            self.failure_csv.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # public API – unchanged from old code
    # ------------------------------------------------------------------ #
    def record(
            self,
            md5: str,
            model_path: str,
            img_path: str,
            bounds: Any,
            ratio: Optional[float],
            dark_bright_value: Optional[float],
            cam_k: Any,
            lgt_k: Any,
            pose_k: Any,
            mat_k: Any | None,
            error: Optional[str] = None,
    ) -> None:
        """
        One-shot helper the existing code calls.
        * If `error` is None  -> success row.
        * Else                -> failure row.
        """
        row = {
            "md5": md5,
            "model": model_path,
            "image": img_path,
            "bounds": bounds,
            "ratio": ratio,
            "dark_bright_value": dark_bright_value,
            "cam": cam_k,
            "light": lgt_k,
            "pose": pose_k,
            "mat": mat_k,
            "error": error,
        }
        if self.add_timestamp:
            row["timestamp"] = datetime.utcnow().isoformat(timespec="seconds")

        success = error is None
        self._enqueue(row, success=success)

        if success:
            # maintain per-md5 counts _and_ path sets
            self._success_counts[md5] += 1
            self._success_paths[md5].add(img_path)
            self._all_success_paths.add(img_path)

    def has_all(self, md5: str, expected: int) -> bool:
        """True if we already logged ≥ `expected` successful rows for this md5."""
        return self._success_counts.get(md5, 0) >= expected

    def has(self, img_path: Path) -> bool:
        """
        Return True if *this exact* image path was already logged
        successfully in the current or a previous session.
        """
        return str(img_path) in self._all_success_paths

    def close(self) -> None:
        """Flush buffers to CSV."""
        self._flush(success=True)
        self._flush(success=False)

    # enable `with RenderLogger(...) as log: ...`
    def __enter__(self) -> "RenderLogger":
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        self.close()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _load_success_counts(self) -> None:
        """
        Populate _success_counts, _success_paths and _all_success_paths
        from an existing success CSV (if present).
        """
        if not self.success_csv.exists():
            return

        with self.success_csv.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            has_header = header and "md5" in header[0].lower()
            if not has_header:  # rewind if no header detected
                f.seek(0)
                reader = csv.reader(f)

            for row in reader:
                if not row:
                    continue
                md5 = row[0]
                # image column is the third position in the current schema
                img_path = row[2] if len(row) > 2 else ""
                if img_path and Path(img_path).exists():
                    self._success_counts[md5] += 1
                    self._success_paths[md5].add(img_path)
                    self._all_success_paths.add(img_path)

    def _enqueue(self, row: Dict[str, Any], *, success: bool) -> None:
        buf = self._succ_buffer if success else self._fail_buffer
        buf.append(row)
        if len(buf) >= self.chunk_size:
            self._flush(success=success)

    def _flush(self, *, success: bool) -> None:
        buf = self._succ_buffer if success else self._fail_buffer
        if not buf:
            return

        dest = self.success_csv if success else self.failure_csv
        if dest is None:
            buf.clear()
            return

        header_flag = "_succ_header_written" if success else "_fail_header_written"
        header_written = getattr(self, header_flag)

        pd.DataFrame(buf).to_csv(
            dest,
            mode="a",
            header=not header_written,
            index=False,
        )

        setattr(self, header_flag, True)
        buf.clear()

    def __del__(self) -> None:  # noqa: D401
        try:
            self.close()
        except Exception:
            pass
