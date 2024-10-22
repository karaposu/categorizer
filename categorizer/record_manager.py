from doctest import debug

# this is record_manager.py
import pandas as pd
from llmec.record import Record
from llmec.categorization_engine import  CategorizationEngine
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
from llmec.metapattern_manager import MetaPatternManager


class RecordManager:
    def __init__(self, debug=False):
         
        self.records = []
        self.cache = {}
        self.cache_fill_count = 0
        self.debug=debug


        self._debug(f"RecordManager being initialized")

        self.categorization_engine = CategorizationEngine(subcategory_level=2)


        meta_patterns_yaml_path= 'llmec/bank_patterns.yaml'

        self.mpm = MetaPatternManager(meta_patterns_yaml_path)

        self._debug(f"initialization finished")



        # self.mpm.loaded_yaml_data

        # meta_patterns = self.loaded_yaml_data[meta_pattern_owner]

    def _debug(self, message):
        if self.debug:
            logger.debug(message)

    def categorize_records(self):
        all_dfs = []
        for r in self.records:
            self.categorize_a_record(r)
            all_dfs.append(r.df)

        df = pd.concat(all_dfs, ignore_index=True)

        return df

            # elapsed_time_in_seconds = progress_bar.format_dict['elapsed']
            #
            # elapsed_str = progress_bar.format_interval(elapsed_time_in_seconds)
            # if db_manager:
            #     db_manager.advanced_log_progress(i, total_records, user_id, file_id, elapsed_time_in_seconds, 1)

    def categorize_lvl_by_lvl(self, record, use_metapattern=False, use_keyword=False):


        self.categorization_engine.categorize_lvl_by_lvl(record)




    def new_categorize_a_record(self,
                            record,
                            use_metapattern=False,
                            use_keyword=False,
                            use_cache=False):

        self._debug(f"categorize_a_record()")

        if use_cache:
            self.fill_from_cache_using_keywords(record)

        # if use_metapattern:
        #     # if record.metapatterns["auto_categorization_patterns"] is not None:
        #         record.categorize_with_metapattern()


        self.categorization_engine.categorize_record(record)

        if use_metapattern:
            if record.metapatterns["classification_patterns"] is not None:
                 record.categorize_with_metapattern()

        if use_keyword:
                record.categorize_with_auto_trigger_keyword()

        if not record.ready:
            record.categorize()

        # self.update_cache()
        logger.debug("categorize Done")


    def categorize_a_record(self,
                            record,
                            use_metapattern=False,
                            use_keyword=False,
                            use_cache=False):

        self._debug(f"categorize_a_record()")

        if use_cache:
            self.fill_from_cache_using_keywords(record)

        if use_metapattern:
                record.categorize_with_metapattern()

        if use_keyword:
                record.categorize_with_auto_trigger_keyword()


        if not record.ready:
            self.categorization_engine.categorize_lvl_by_lvl(record)


        logger.debug("categorize Done")




    def naive_categorization(self):

        logger.debug("naive_classification()", extra={'lvl': 1})

        for r in tqdm(self.records):
            classification_patterns = self.mpm.bring_specific_meta_pattern(r.associated_with, "classification_patterns")

            self.categorization_engine.categorize_record_with_meta_pattern(r, classification_patterns)

            if r.categorized_by is None:
                self.categorization_engine.categorize_record_with_keyword(r)

    def cache_results(self, record):
        keyword = record.keyword
        self.cache[keyword] = record.clone()
        # if keyword and record.ready:  # Ensure record is complete before caching
        #     self.cache[keyword] = record.clone()  # Store a copy of the record to avoid mutation issues
        #     logger.debug(f"Cached results for keyword: {keyword}")

    def fill_from_cache_using_keywords(self, record):
        keyword = record.keyword
        self._debug(f"fill_from_cache_using_keywords()")
        self._debug(f"keyword: {keyword}")
        if keyword is not None:
            if keyword in self.cache:
                self._debug(f"keyword found in cache")
                cached_record = self.cache[keyword]
                #logger.debug("cached_record.rationale =%s ", cached_record.rationale, extra={'lvl': 4})
                record.apply_cached_result(cached_record)  # Update the record with cached results
                self.cache_fill_count += 1  # Increment the counter
                return True
            return False
        return False

    def get_number_of_ready_records(self):
        # Count and return the number of records that are marked as 'ready'
        return sum(1 for record in self.records if record.ready)

    def load_records(self, record_inputs, categories_yaml_path='categories.yaml', record_ids=None):
        if isinstance(record_inputs, pd.DataFrame):

            self._load_from_dataframe(record_inputs, categories_yaml_path)
        elif isinstance(record_inputs, list):
            self._load_from_list(record_inputs, record_ids, categories_yaml_path)
        elif isinstance(record_inputs, str):
            self._load_from_string(record_inputs, record_ids, categories_yaml_path)
        else:
            raise ValueError("Input should be a pandas DataFrame, list, or string")


        for r in self.records:
            r.metapatterns= self.mpm.loaded_yaml_data[r.associated_with]

        self._debug(f"records are loaded. Number of records: {len(self.records)}")





    def _load_from_dataframe(self, df: pd.DataFrame, categories_yaml_path: str):
        #logger.debug("Loading records from DataFrame")
        for _, row in df.iterrows():
            record = Record(text=pd.DataFrame([row]), categories=categories_yaml_path, logger=logger)

            self.records.append(record)

    def _load_from_list(self, record_inputs: list, record_ids, categories_yaml_path: str):
        logger.debug("Loading records from list")
        ids = record_ids or [None] * len(record_inputs)
        for idx, text in enumerate(record_inputs):
            record = Record(text=text, record_id=ids[idx], categories=categories_yaml_path)
            self.records.append(record)

    def _load_from_string(self, record_input: str, record_id, categories_yaml_path: str):
        logger.debug("Loading record from string")
        record = Record(text=record_input, record_id=record_id, categories=categories_yaml_path)
        self.records.append(record)



def main():
    import logging
    import sys
    from indented_logger import setup_logging, smart_indent_log
    setup_logging(level=logging.DEBUG, include_func=True)

    rm=RecordManager(debug=True)

    import pandas as pd
    import yaml

    sample_records_path= "llmec/sample_records.yaml"

    # Load the YAML file
    with open(sample_records_path, 'r') as file:
        data = yaml.safe_load(file)

    # Extract records and convert to a DataFrame
    records = data.get('records', [])
    df = pd.DataFrame(records)

    # print(df)

    # dummy_df = pd.DataFrame(data)

    rm.load_records(df, categories_yaml_path='llmec/categories.yaml')

    one=rm.records[0]

    result_df=rm.categorize_records()
    print(result_df)

   # rm.categorize_a_record(one, use_metapattern=False, use_keyword=False)




if __name__ == '__main__':
    main()