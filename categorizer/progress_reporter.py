# categorizer/progress_reporter.py

import logging
import time
from datetime import timedelta
from typing import Protocol, Optional

class ProgressReporter(Protocol):
    def update_processing_status(
        self,
        process_status:   Optional[str]  = None,  # e.g. "started", "progress", "completed"
        processed:        Optional[int]  = None,  # how many done so far
        total:            Optional[int]  = None,  # total to do
        failed_count:     Optional[int]  = None,  # how many have failed
        start_time:       Optional[float]= None,  # timestamp when processing began
    ) -> None:
        ...


class LogReporter(ProgressReporter):
    """
    Logs a unified status update whenever update_processing_status is called.
    `process_status` drives whether this is a start, a progress tick, or completion.
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger    = logger or logging.getLogger(__name__)
        self._start_ts = None
        self._total    = None

    def update_processing_status(
        self,
        process_status:   Optional[str]   = None,
        processed:        Optional[int]   = None,
        total:            Optional[int]   = None,
        failed_count:     Optional[int]   = None,
        start_time:       Optional[float] = None,
    ) -> None:
        now = time.time()

        # ---- Handle “started” ----
        if process_status == "started":
            # record start timestamp and total
            self._start_ts = now
            self._total    = total or 0
            self.logger.info(f"[Reporter] Starting processing of {self._total} records.")
            return

        # ---- Handle “progress” ----
        if process_status == "progress" and processed is not None and total is not None and start_time is not None:
            elapsed = now - start_time
            pct     = (processed / total) * 100 if total else 0
            # ETA calculation
            if processed > 0:
                rate      = elapsed / processed
                remaining = rate * (total - processed)
                eta       = timedelta(seconds=int(remaining))
            else:
                eta = "N/A"
            self.logger.info(
                f"[Reporter] {processed}/{total} "
                f"({pct:.1f}%), elapsed {elapsed:.1f}s, ETA {eta}"
            )
            return

        # ---- Handle “completed” ----
        if process_status == "completed":
            st = start_time or self._start_ts or now
            elapsed = now - st
            self.logger.info(
                f"[Reporter] Finished {total} records in {elapsed:.1f}s "
                f"with {failed_count or 0} failures."
            )
            return

        # ---- Fallback / custom status ----
        # show whatever fields were passed
        parts = []
        if process_status:
            parts.append(f"Status={process_status}")
        if processed is not None and total is not None:
            parts.append(f"{processed}/{total}")
        if failed_count is not None:
            parts.append(f"failures={failed_count}")
        msg = "[Reporter] " + " ".join(parts) if parts else "[Reporter] update"
        self.logger.info(msg)
