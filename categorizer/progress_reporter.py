# progress_reporter.py

import logging
import time
from datetime import timedelta
from typing import Protocol


class ProgressReporter(Protocol):
    """
    Protocol for reporting progress during long‑running tasks.
    Any concrete reporter must implement these methods.
    """
    def start(self, total_records: int) -> None:
        ...

    def log_zero_progress(self) -> None:
        ...

    def update_processing_status(self, new_status) -> None:
        ...

    def log_progress(
        self,
        number_of_processed_records: int,
        start_time: float,
        total_records: int
    ) -> None:
        ...

    def finish(self, total: int, failed_count: int) -> None:
        ...


class LogReporter(ProgressReporter):
    """
    A concrete ProgressReporter that logs progress, ETA, and status updates.
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.total = 0
        self.start_ts = 0.0

    def start(self, total_records: int) -> None:
        self.total = total_records
        self.start_ts = time.time()
        self.logger.info(f"[Reporter] Starting processing of {self.total} records.")

    def log_zero_progress(self) -> None:
        self.logger.info(f"[Reporter] Progress: 0/{self.total} (0.0%)")

    def update_processing_status(self, new_status) -> None:
        self.logger.info(f"[Reporter] Status update: {new_status}")

    def log_progress(
        self,
        number_of_processed_records: int,
        start_time: float,
        total_records: int
    ) -> None:
        elapsed = time.time() - start_time
        pct = (number_of_processed_records / total_records) * 100
        if number_of_processed_records > 0:
            rate = elapsed / number_of_processed_records
            remaining = rate * (total_records - number_of_processed_records)
            eta = timedelta(seconds=int(remaining))
        else:
            eta = "N/A"
        self.logger.info(
            f"[Reporter] {number_of_processed_records}/{total_records} "
            f"({pct:.1f}%), elapsed: {elapsed:.1f}s, ETA: {eta}"
        )

    def finish(self, total: int, failed_count: int) -> None:
        elapsed = time.time() - self.start_ts
        self.logger.info(
            f"[Reporter] Finished {total} records in {elapsed:.1f}s "
            f"with {failed_count} failures."
        )
