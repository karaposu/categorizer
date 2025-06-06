# categorizer/progress_reporter.py

import logging
from datetime import datetime, timezone, timedelta
from typing import Protocol


class ProgressReporter(Protocol):
    """
    Protocol defining granular progress-reporting methods.
    """
    def update_status(self, status: str) -> None:
        """e.g. "started", "in_progress", "completed", or custom."""
        ...

    def update_total(self, total: int) -> None:
        """Set or update the total number of items."""
        ...

    def update_processed_count(self, processed: int) -> None:
        """Report how many items have been processed so far."""
        ...

    def update_failed_count(self, failed: int) -> None:
        """Report how many failures have occurred so far."""
        ...

    def update_percentage(self, percentage: float) -> None:
        """Report progress as a percentage (0.0–100.0)."""
        ...

    def update_remaining_time(self, remaining_time_seconds: float) -> None:
        """Report estimated remaining time in seconds."""
        ...


class LogReporter(ProgressReporter):
    """
    Concrete logger-based reporter implementing the above methods.
    Uses UTC datetime for timestamps.
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger     = logger or logging.getLogger(__name__)
        

        self.start_ts=None
        self.finished_ts=None
        self._total     = 0
        self._processed = 0
        self._failed    = 0


    def update_status(self, status: str) -> None:
        self.logger.info(f"[Reporter] Status: {status}")
        if status == "started":
            self.start_ts = datetime.now(timezone.utc)
            self.logger.info(f"[Reporter] start time: { self.start_ts}")
           
        elif status == "completed":
            self.finished_ts = datetime.now(timezone.utc)
           
        # self.logger.info(f"[Reporter] Status: {status}")

   
    def update_total(self, total: int) -> None:
        self._total = total
        self.logger.info(f"[Reporter] Total items to process: {total}")

    def update_processed_count(self, processed: int) -> None:
        self._processed = processed
        self.logger.info(f"[Reporter] Processed: {processed}")

    def update_failed_count(self, failed: int) -> None:
        self._failed = failed
        self.logger.info(f"[Reporter] Failed: {failed}")

    def update_percentage(self, percentage: float) -> None:
        self.logger.info(f"[Reporter] Progress: {percentage:.1f}%")

    def update_remaining_time(self, remaining_time_seconds: float) -> None:

        if remaining_time_seconds is None:
            remaining_time_seconds=0
        eta = timedelta(seconds=int(remaining_time_seconds))
        self.logger.info(f"[Reporter] ETA: {eta}")

    @staticmethod
    def find_remaining_time(
        start_time: datetime,
        total_records: int,
        processed: int
    ) :
        """
        Linear ETA: (elapsed / processed) * (total - processed).
        Returns remaining seconds, or None if cannot compute.
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - start_time).total_seconds()
        if processed <= 0 or processed >= total_records:
            return None
        rate = elapsed / processed
        return int(rate * (total_records - processed))

