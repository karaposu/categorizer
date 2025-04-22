# record_manager.py

import pandas as pd
from categorizer.record import Record
from categorizer.categorization_engine import CategorizationEngine
import logging
from tqdm import tqdm
from categorizer.metapattern_manager import MetaPatternManager
from indented_logger import setup_logging, log_indent

from concurrent.futures import ThreadPoolExecutor, as_completed

from categorizer.progress_reporter import ProgressReporter, LogReporter


from typing import Optional
from time import time
from datetime import datetime

# Configure the logger
logger = logging.getLogger(__name__)

#python -m categorizer.record_manager


class RecordManager:
    def __init__(self, debug=False):
        self.records = []
        self.debug = debug
        self.logger = logger
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        logger.debug("RecordManager being initialized")

        # Initialize the categorization engine
        self.categorization_engine = CategorizationEngine(debug=True)
        self.cache={}

        # Initialize MetaPatternManager if needed
        meta_patterns_yaml_path = 'categorizer/bank_patterns.yaml'
        self.mpm = MetaPatternManager(meta_patterns_yaml_path)

       # self._debug("Initialization finished")

    def _debug(self, message):
        if self.debug:
            self.logger.debug(message)

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

    def _load_from_list(self, record_inputs: list, record_ids, categories_yaml_path: str):
        #self.logger.debug("Loading records from list")
        ids = record_ids or [None] * len(record_inputs)
        for idx, text in enumerate(record_inputs):
            record = Record.from_string(text=text, record_id=ids[idx], categories=categories_yaml_path)
            self.records.append(record)
    
    def _load_from_string(self, record_input: str, record_id, categories_yaml_path: str):
       # self.logger.debug("Loading record from string")
        record = Record.from_string(text=record_input, record_id=record_id, categories=categories_yaml_path)
        self.records.append(record)

    def _categorize_in_batches(
        self,
        batch_size: int,
        reporter: Optional[ProgressReporter],
        start_ts: float,
        total: int
    ) -> int:
        """
        Process records in batches of `batch_size`, using a ThreadPoolExecutor.
        Reports progress after each record completes.
        Returns number of failures.
        """
        failures = 0
        processed = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = self.records[start:end]
            logger.debug(f"Processing batch {start+1}–{end} of {total}")

            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(
                        self.categorize_a_record,
                        rec,
                        False,  # use_metapattern
                        True,   # use_keyword
                        True    # use_cache
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
                        logger.error(
                            f"Error categorizing record {rec.record_id}: {e}",
                            exc_info=True
                        )
                    finally:
                        if reporter:
                            reporter.log_progress(
                                number_of_processed_records=processed,
                                start_time=start_ts,
                                total_records=total
                            )

        return failures
    
    def _categorize_one_by_one(
        self,
        reporter: Optional[ProgressReporter],
        start_ts: float,
        total: int
    ) -> int:
        """
        Serial processing, reporting after each record.
        Returns number of failures.
        """
        failures = 0

        for idx, rec in enumerate(self.records, start=1):
            try:
                self.categorize_a_record(
                    rec,
                    use_metapattern=False,
                    use_keyword=True,
                    use_cache=True
                )
            except Exception:
                failures += 1
                logger.error(f"Error categorizing record {rec.record_id}: {Exception}", exc_info=True)
            finally:
                if reporter:
                    reporter.log_progress(
                        number_of_processed_records=idx,
                        start_time=start_ts,
                        total_records=total
                    )

        return failures
    
    @log_indent
    def categorize_records(
        self,
        batch_size: Optional[int] = None,
        reporter: Optional[ProgressReporter] = None
    ) -> pd.DataFrame:
        """
        Categorize all records, either in batches (threaded) if batch_size is set,
        or one by one otherwise. Progress is reported via the single
        update_processing_status(...) method.
        """
        total = len(self.records)
        logger.debug(f"Starting categorization: {total} record(s)")

        # Empty case
        if total == 0:
            logger.warning("No records to process.")
            if reporter:
                reporter.update_processing_status(
                    process_status="completed",
                    processed=0,
                    total=0,
                    failed_count=0,
                    start_time=time()
                )
            return self.get_records_dataframe()

        # Record start
        start_ts = time()
        if reporter:
            reporter.update_processing_status(
                process_status="started",
                total=total,
                start_time=start_ts
            )

        failures = 0
        processed = 0

        # --- BATCHED PATH ---
        if batch_size:
            for offset in range(0, total, batch_size):
                batch = self.records[offset : offset + batch_size]
                logger.debug(f"Processing batch {offset+1}–{offset+len(batch)} of {total}")

                with ThreadPoolExecutor(max_workers=len(batch)) as execr:
                    futures = {
                        execr.submit(
                            self.categorize_a_record,
                            rec,
                            False,  # use_metapattern
                            True,   # use_keyword
                            True    # use_cache
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
                            logger.error(
                                f"Error categorizing record {rec.record_id}: {e}",
                                exc_info=True
                            )
                        finally:
                            if reporter:
                                reporter.update_processing_status(
                                    process_status="progress",
                                    processed=processed,
                                    total=total,
                                    start_time=start_ts
                                )

        # --- SERIAL PATH ---
        else:
            for rec in self.records:
                processed += 1
                try:
                    self.categorize_a_record(
                        rec,
                        use_metapattern=False,
                        use_keyword=True,
                        use_cache=True
                    )
                except Exception as e:
                    failures += 1
                    logger.error(
                        f"Error categorizing record {rec.record_id}: {e}",
                        exc_info=True
                    )
                finally:
                    if reporter:
                        reporter.update_processing_status(
                            process_status="progress",
                            processed=processed,
                            total=total,
                            start_time=start_ts
                        )

        # --- FINISH ---
        if reporter:
            reporter.update_processing_status(
                process_status="completed",
                processed=processed,
                total=total,
                failed_count=failures,
                start_time=start_ts
            )

        return self.get_records_dataframe()


    # @log_indent
    # def categorize_records(
    #     self,
    #     batch_size: Optional[int] = None,
    #     reporter: Optional[ProgressReporter] = None
    # ) -> pd.DataFrame:
    #     """
    #     Categorize all records, either in batches (threaded) if batch_size is set,
    #     or one by one otherwise.  Progress is reported via `reporter`.
    #     """
    #     total = len(self.records)
    #     logger.debug(f"Starting categorization: {total} record(s)")

    #     # Empty case
    #     if total == 0:
    #         logger.warning("No records to process.")
    #         if reporter:
    #             reporter.update_processing_status(new_status="completed")
    #         return self.get_records_dataframe()

    #     # Kick off reporter
    #     if reporter:
    #         reporter.start(total)
    #         reporter.log_zero_progress()

    #     start_ts = time()
    #     # Choose path
    #     if batch_size:
    #         failures = self._categorize_in_batches(
    #             batch_size=batch_size,
    #             reporter=reporter,
    #             start_ts=start_ts,
    #             total=total
    #         )
    #     else:
    #         failures = self._categorize_one_by_one(
    #             reporter=reporter,
    #             start_ts=start_ts,
    #             total=total
    #         )

    #     # Finish reporter
    #     if reporter:
    #         reporter.finish(total=total, failed_count=failures)

    #     return self.get_records_dataframe()
    
    

    @log_indent
    def categorize_a_record(self, record, use_metapattern=False, use_keyword=False, use_cache=False):
        logger.debug(f"Categorizing record.id: {record.record_id}, record.text {record.text}" )

        if use_cache:
            self.fill_from_cache_using_keywords(record)

        if not record.ready:
            self.categorization_engine.categorize_record(record, use_metapattern=use_metapattern, use_keyword=use_keyword)
            self.cache_results(record)


        logger.debug(f"record dict: {record.to_dict()}")

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

    def get_records_dataframe(self):
        records_data = [record.to_dict() for record in self.records]
        df = pd.DataFrame(records_data)
        return df

    def get_number_of_ready_records(self):
        return sum(1 for record in self.records if record.ready)

    # Cache functionality (if needed)
    def cache_results(self, record):
        keyword = record.keyword
        if keyword and record.ready:
            self.cache[keyword] = record.clone()
            self.logger.debug(f"Cached results for keyword: {keyword}")

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
    
    # def categorize_records_in_batches(
    #     self,
    #     record_batch_size: int,
    #     use_metapattern: bool = False,
    #     use_keyword: bool = False,
    #     use_cache: bool = False,
    # ) -> pd.DataFrame:
    #     """
    #     Loop over all records in batches, delegating each batch to _process_batch().
    #     """
    #     total = len(self.records)
    #     for start in range(0, total, record_batch_size):
    #         end = min(start + record_batch_size, total)
    #         batch = self.records[start:end]
    #         self._process_batch(batch, start, end, total,
    #                             use_metapattern, use_keyword, use_cache)

    #     return self.get_records_dataframe()


    # def _process_batch(
    #     self,
    #     batch: list,
    #     start: int,
    #     end: int,
    #     total: int,
    #     use_metapattern: bool,
    #     use_keyword: bool,
    #     use_cache: bool,
    # ) -> None:
    #     """
    #     Categorize a single batch of records in parallel using threads.
    #     """
    #     logger = logging.getLogger(__name__)
    #     logger.debug(f"Processing records {start}–{end-1} of {total}")

    #     # For I/O‑bound OpenAI calls, threads give high concurrency with low overhead
    #     with ThreadPoolExecutor(max_workers=len(batch)) as executor:
    #         # submit each record to its own thread
    #         futures = {
    #             executor.submit(
    #                 self.categorize_a_record,
    #                 record,
    #                 use_metapattern,
    #                 use_keyword,
    #                 use_cache
    #             ): record
    #             for record in batch
    #         }

    #         # wait for completion and handle per‐record errors
    #         for future in as_completed(futures):
    #             record = futures[future]
    #             try:
    #                 future.result()
    #             except Exception as e:
    #                 logger.error(
    #                     f"Error categorizing record {record.record_id}: {e}",
    #                     exc_info=True
    #                 )


def main():
    from indented_logger import setup_logging, log_indent
  
    setup_logging(level=logging.DEBUG, include_func=True)
    
    rm = RecordManager(debug=True)

    import yaml

    sample_records_path = "categorizer/sample_records.yaml"

    # Load the YAML file
    with open(sample_records_path, 'r') as file:
        data = yaml.safe_load(file)

    # Extract records and convert to a DataFrame
    records = data.get('records', [])
    df = pd.DataFrame(records)

    # Load records into RecordManager
    rm.load_records(df, categories_yaml_path='categorizer/categories.yaml')
    
    # Categorize records
    t0=time()

    
    reporter = LogReporter()
    
    result_df = rm.categorize_records(reporter=reporter)
    # result_df = rm.categorize_records(batch_size=7, reporter=reporter)

    t1=time()

    # Print the resulting DataFrame
    print("Categorization results:")
    print(result_df)
    
    print("total_time: ", t1-t0)


if __name__ == '__main__':
    main()
