# categorization_engine.py

import yaml
import logging
import re
from time import time

from indented_logger import setup_logging, log_indent, smart_indent_log
from .schemas import CategorizationResult
from categorizer.myllmservice import MyLLMService

logger = logging.getLogger(__name__)

class CategorizationEngine:
    def __init__(self, subcategory_level=1, debug=False):
        self.subcategory_level = subcategory_level
        self.use_system_prompt = False
        self.debug = debug

        self.myllmservice = MyLLMService()
        self.logger = logging.getLogger(__name__)
        self.llm_system_prompt = ""

        self.classified_by_meta_pattern_count = 0
        self.classified_by_keyword_count = 0

    def _debug(self, message):
        if self.debug:
            self.logger.debug(message)

    @log_indent
    def categorize_record(self, record, use_metapattern=False, use_keyword=False):

        if use_metapattern:
            self.categorize_with_metapattern(record)

        if use_keyword:
            self.categorize_with_auto_trigger_keyword(record)

        if not record.ready:
            self.categorize_lvl_by_lvl(record)



    def categorize_with_metapattern(self, record):
        if record.metapatterns and record.metapatterns.get("classification_patterns"):
            # self._debug("Inside categorize_with_metapattern")
            logger.debug("Inside categorize_with_metapattern")

            classification_patterns = record.metapatterns["classification_patterns"]
            categorization_result = self.categorize_record_with_meta_pattern(
                record.text, classification_patterns
            )

            matched_pattern = categorization_result.matched_pattern
            logger.debug(f"Matched pattern: {matched_pattern}")

            if categorization_result.success:
                self._select_categories_from_pattern(record, matched_pattern)
                record.ready = True
                record.categorized_by = "metapattern"
                record.generate_merged_category_dict()
                record.rationale = "metapattern"
                record.refiner_output = "n"
                # Removed call to record.generate_df()

    def categorize_with_auto_trigger_keyword(self, record):
        categorization_result = self.categorize_record_with_keyword(record.available_categories, record.text)
        if categorization_result.success:
            matched_keyword = categorization_result.matched_keyword
            category_list = categorization_result.category_list
            for cat in category_list:
                record.select_lvl_category(cat.lvl, cat.name, classified_by="keyword")
            record.ready = True
            record.categorized_by = "keyword"
            record.generate_merged_category_dict()
            record.rationale = ""
            record.refiner_output = "n"
            # Removed call to record.generate_df()

    def prepare_category_documentation(self, record, level):
        """
        Prepares the merged category documentation for the given level.

        Args:
            record (Record): The record being categorized.
            level (int): The current level of categorization.

        Returns:
            tuple: A tuple containing a boolean indicating success, and the merged category documentation or None.
        """

        if level == 1:
            lvl1_categories = record.filter_categories_by_lvl(level)
            self.logger.debug(f"Number of level 1 categories: {len(lvl1_categories)}")
            docs = record.category_list_to_docs(lvl1_categories)
            merged_cat_doc = " ".join(docs)
            return True, merged_cat_doc
        else:
            parent_category = record.get_parent_category_of_lvl(level)
            if parent_category:
                valid_categories = parent_category.child_categories
                docs = record.category_list_to_docs(valid_categories)
                merged_cat_doc = " ".join(docs)
                return True, merged_cat_doc
            else:
                self.logger.debug(f"No parent category at level {level}")
                return False, None

    @log_indent
    def categorize_level(self, record, level, merged_cat_doc):
        """
        Performs categorization for the given level and processes the result.

        Args:
            record (Record): The record being categorized.
            level (int): The current level of categorization.
            merged_cat_doc (str): The merged category documentation for the level.

        Returns:
            bool: True if categorization was successful, False otherwise.
        """
        generation_result = self.categorize_text_with_LLM(record.text, merged_cat_doc)
        self.logger.debug(f"LLM result at level {level}: {generation_result.content}")

        if generation_result.success:
            if record.validate_proposed_category_and_hierarchy(generation_result.content, level):
                record.select_lvl_category(level, generation_result.content)
                record.fill_rationale(level=level, value=generation_result.raw_content)
                record.fill_refiner_output(level=level, value=generation_result.content)
                record.categorized_by = "llm"
                return True
            else:
                self.logger.debug(f"Category is not valid at level {level}")
                record.select_lvl_category(level, "cant_determined")
                return False
        else:
            self.logger.debug(f"Categorization failed at level {level}")
            record.select_lvl_category(level, "cant_determined")
            return False

    @log_indent
    def categorize_lvl_by_lvl(self, record):
        sequential_success = True
        for level in range(1, record.depth + 1):
            self.logger.debug(f"Current level is {level}")

            # Preparation Phase
            prep_success, merged_cat_doc = self.prepare_category_documentation(record, level)
            if not prep_success:
                sequential_success = False
                record.select_lvl_category(level, "cant_determined")
                break

            # Categorization Phase
            categorization_success = self.categorize_level(record, level, merged_cat_doc)
            if not categorization_success:
                sequential_success = False
                break

        if sequential_success:
            record.ready = True




    def categorize_record_with_meta_pattern(self, text, classification_patterns):
        matched_pattern = next(
            (pattern for pattern in classification_patterns if re.search(pattern['pattern'], text)), None
        )

        categorization_result = CategorizationResult(
            success=False,
            category_list=[],
            rationale_dict={},
            matched_pattern=None,
            raw_llm_answer=None,
            matched_keyword=None,
            categorized_by=None
        )
        if matched_pattern:
            categorization_result.success = True
            categorization_result.matched_pattern = matched_pattern
            categorization_result.categorized_by = "metapattern"

        return categorization_result

    def check_keywords_in_category(self, cat, text):
        for k in cat.auto_trigger_keyword:
            if k.lower() in text.lower():
                return k, cat
        return None

    def categorize_record_with_keyword(self, available_categories, text):
        categorization_result = CategorizationResult(
            success=False,
            category_list=[],
            rationale_dict={},
            matched_pattern=None,
            raw_llm_answer=None,
            matched_keyword=None,
            categorized_by=None
        )

        for cat in available_categories:
            result = self.check_keywords_in_category(cat, text)
            if result:
                matched_keyword, matched_cat = result
                parent_cat = matched_cat.parent_categories[0]
                categorization_result.success = True
                categorization_result.matched_keyword = matched_keyword
                categorization_result.categorized_by = "keyword"
                categorization_result.category_list.append(parent_cat)
                categorization_result.category_list.append(matched_cat)
                break

        return categorization_result

    @log_indent
    def categorize_text_with_LLM(self, text, classes):
        class ExpandedDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        classes_string = yaml.dump(classes, default_flow_style=False, Dumper=ExpandedDumper)
        generation_result = self.myllmservice.categorize_simple(text, classes_string)
        return generation_result

    def _select_categories_from_pattern(self, record, pattern):
        record.select_lvl_category(1, pattern['lvl1'], classified_by="metapattern")
        for level in range(2, self.subcategory_level + 1):
            lvl_key = f'lvl{level}'
            if lvl_key in pattern:
                record.select_lvl_category(level, pattern[lvl_key], classified_by="metapattern")

def main():
    from indented_logger import setup_logging, log_indent, smart_indent_log

    setup_logging(
        level=logging.DEBUG,
        include_func=True,
        truncate_messages=False,
        min_func_name_col=100,
        include_module=True,
        indent_modules=True,
    )

    # Initialize CategorizationEngine
    categorization_engine = CategorizationEngine(subcategory_level=2, debug=True)

    # Sample data for testing
    target_string = "The company reported a significant increase in revenue this quarter."

    categories_and_helpers = [
        {
            'category': 'Finance',
        },
        {
            'category': 'Business',
        }
    ]

    # Perform categorization
    generation_result = categorization_engine.categorize_text_with_LLM(target_string, categories_and_helpers)

    # Print the generation result
    smart_indent_log(logger, generation_result, lvl=2, name="generation_result", flatten_long_strings=True)

if __name__ == '__main__':
    main()
