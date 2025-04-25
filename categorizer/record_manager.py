# categorizer/record_manager.py

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from typing import Optional

import pandas as pd
from indented_logger import log_indent

from categorizer.record import Record
from categorizer.categorization_engine import CategorizationEngine
from categorizer.metapattern_manager import MetaPatternManager
from categorizer.progress_reporter import ProgressReporter, LogReporter

logger = logging.getLogger(__name__)

#python -m categorizer.record_manager


class RecordManager:
    def __init__(self, debug: bool = False):
        self.cache = {}
        self.records = []
        self.debug = debug
        self.logger = logger

        self.total=None

        self._reporter= None
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        logger.debug("RecordManager being initialized")
        self.categorization_engine = CategorizationEngine(debug=True)
        self.mpm = MetaPatternManager('categorizer/bank_patterns.yaml')

    
    def fill_from_cache_using_keywords(self, record):
        keyword = record.keyword
        logger.debug(f"Checking cache for keyword: {keyword}")
        if keyword is not None:
            if keyword in self.cache:
                logger.debug("Keyword found in cache")
                cached_record = self.cache[keyword]
                record.apply_cached_result(cached_record)
                return True
        return False
    
    def cache_results(self, record):
        keyword = record.keyword
        if keyword and record.ready:
            self.cache[keyword] = record.clone()
            self.logger.debug(f"Cached results for keyword: {keyword}")

    @log_indent
    def categorize_a_record(self, record, use_metapattern=False, use_keyword=False, use_cache=False):
        logger.debug(f"Categorizing record.id: {record.record_id}, record.text {record.text}" )

        if use_cache:
            self.fill_from_cache_using_keywords(record)

        if not record.ready:
            self.categorization_engine.categorize_record(record, use_metapattern=use_metapattern, use_keyword=use_keyword)
            self.cache_results(record)


        logger.debug(f"record dict: {record.to_dict()}")


    @log_indent
    def load_records(self, record_inputs, categories_yaml_path='categories.yaml', record_ids=None):
        if isinstance(record_inputs, pd.DataFrame):
            self._load_from_dataframe(record_inputs, categories_yaml_path)
        elif isinstance(record_inputs, list):
            self._load_from_list(record_inputs, record_ids, categories_yaml_path)
        elif isinstance(record_inputs, str):
            self._load_from_string(record_inputs, record_ids, categories_yaml_path)
        else:
            raise ValueError("Input should be a pandas DataFrame, list, or string")

        # Assign metapatterns to records if needed
        for r in self.records:
            if r.associated_with:
                r.metapatterns = self.mpm.loaded_yaml_data.get(r.associated_with, {})
            else:
                r.metapatterns = {}
        logger.debug(f"Records are loaded. Number of records: {len(self.records)}")

    def _load_from_dataframe(self, df: pd.DataFrame, categories_yaml_path: str):
       # self.logger.debug("Loading records from DataFrame")
        for _, row in df.iterrows():
            record = Record.from_dataframe(row, categories=categories_yaml_path, logger=self.logger)
            self.records.append(record)
            self.update_total(len(self.records))

    def _load_from_list(self, record_inputs: list, record_ids, categories_yaml_path: str):
        #self.logger.debug("Loading records from list")
        ids = record_ids or [None] * len(record_inputs)
        for idx, text in enumerate(record_inputs):
            record = Record.from_string(text=text, record_id=ids[idx], categories=categories_yaml_path)
            self.records.append(record)
            self.update_total(len(self.records))
    
    def _load_from_string(self, record_input: str, record_id, categories_yaml_path: str):
       # self.logger.debug("Loading record from string")
        record = Record.from_string(text=record_input, record_id=record_id, categories=categories_yaml_path)
        self.records.append(record)
        self.update_total(len(self.records))

    @log_indent
    def categorize_records(
        self,
        batch_size: Optional[int] = None,
        reporter: Optional[ProgressReporter] = None
    ) -> pd.DataFrame:
       
        logger.debug(f"Starting categorization: {self.total} record(s)")
        total=self.total
        if total == 0:
            return self._handle_empty(reporter)

        start_ts = time()
        if reporter:
            reporter.update_processed_count(0)
            reporter.update_failed_count(0)
            reporter.update_percentage(0.0)
           

        if batch_size:
            failures, processed = self._process_batches(batch_size, reporter, start_ts)
        else:
            failures, processed = self._process_serial(reporter, start_ts)
        
        if reporter:
            self._finalize_reporter(reporter, processed, failures)

        return self.get_records_dataframe()


    def _handle_empty(self, reporter: Optional[ProgressReporter]) -> pd.DataFrame:
        logger.warning("No records to process.")
        if reporter:
            reporter.update_status("completed")
           
            reporter.total= 0
            reporter.update_processed_count(0)
            reporter.update_failed_count(0)
            reporter.update_percentage(100.0)
            reporter.update_remaining_time(0.0)
        return self.get_records_dataframe()

    def update_total(self, total):
        self.total=total


    def _init_reporter(self) -> None:
        reporter = self._reporter
        print(type(reporter))
        reporter.update_processed_count(0)
        reporter.update_failed_count(0)
        reporter.update_percentage(0.0)


    def _process_batches(
        self,
        batch_size: int,
        reporter: Optional[ProgressReporter],
        start_ts: float,
    
    ) -> (int, int):
        failures = 0
        processed = 0
        total=self.total
        for offset in range(0, total, batch_size):
            batch = self.records[offset:offset+batch_size]
            logger.debug(f"Processing batch {offset+1}–{offset+len(batch)} of {total}")

            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(
                        self.categorize_a_record,
                        rec,
                        False, True, True
                    ): rec
                    for rec in batch
                }
                for future in as_completed(futures):
                    processed += 1
                    rec = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        failures += 1
                        logger.error(f"Error categorizing {rec.record_id}: {e}", exc_info=True)
                    finally:
                        if reporter:
                            self._report_progress(reporter, processed, failures, start_ts)

        return failures, processed


    def _process_serial(
        self,
        reporter: Optional[ProgressReporter],
        start_ts: float,
       
    ) -> (int, int):
        failures = 0
        processed = 0
        
        for rec in self.records:
            processed += 1
            try:
                self.categorize_a_record(rec, False, True, True)
            except Exception as e:
                failures += 1
                logger.error(f"Error categorizing {rec.record_id}: {e}", exc_info=True)
            finally:
                if reporter:
                    self._report_progress(reporter, processed, failures, start_ts)

        return failures, processed


    def _report_progress(
        self,
        reporter: ProgressReporter,
        processed: int,
        failures: int,
        start_ts: float
    ) -> None:
        total=self.total
        elapsed = time() - start_ts
        pct     = (processed / total) * 100
        remaining = (elapsed / processed) * (total - processed) if processed else 0.0

        reporter.update_processed_count(processed)
        reporter.update_failed_count(failures)
        reporter.update_percentage(pct)
        reporter.update_remaining_time(remaining)
    
    def get_records_dataframe(self):
        records_data = [record.to_dict() for record in self.records]
        df = pd.DataFrame(records_data)
        return df

    def _finalize_reporter( self, reporter, processed: int,failures: int,) -> None:
       
        reporter.update_status("completed")
        reporter.update_processed_count(processed)
        reporter.update_failed_count(failures)
        reporter.update_percentage(100.0)
        reporter.update_remaining_time(0.0)


def main():
    from indented_logger import setup_logging
    import yaml

    setup_logging(level=logging.DEBUG, include_func=True)
    rm = RecordManager(debug=True)
    
    with open("categorizer/sample_records.yaml") as f:
        data = yaml.safe_load(f)
    df = pd.DataFrame(data["records"])
    rm.load_records(df, categories_yaml_path="categorizer/categories.yaml")
    
    reporter = LogReporter()
    # For serial:
    # result_df = rm.categorize_records(reporter=reporter)
    # For batches:
    result_df = rm.categorize_records(batch_size=5, reporter=reporter)

    print(result_df)


if __name__ == "__main__":
    main()
