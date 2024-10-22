# here is categorization_engine.py


import yaml
import logging
logger = logging.getLogger(__name__)


from indented_logger import setup_logging, log_indent
from .schemas import CategorizationResult



from llmec.myllmservice import MyLLMService

import re
from time import time






class CategorizationEngine:
    def __init__(self,
                 subcategory_level=1,
                 debug=False
                 ):

        self.subcategory_level = subcategory_level
        self.use_system_prompt=False
        self.debug=debug

        self.myllmservice= MyLLMService()


        # self.logger = logger if logger else logging.getLogger(__name__)
        self.logger = logging.getLogger(__name__)

        self.llm_system_prompt= ""

        self.classified_by_meta_pattern_count = 0
        self.classified_by_keyword_count = 0


    def _debug(self, message):
        if self.debug:
            logger.debug(message)


    def categorize_lvl_by_lvl(self, record, use_metapattern=False, use_keyword=False):

        sequential_success = True
        for level in range(1, record.depth + 1):
            self._debug(f"current level is {level}", )
            if level == 1:
                lvl1_categories = record.filter_categories_by_lvl(level)
                self._debug(f"len lvl1_categories {len(lvl1_categories)}", )

                docs = record.category_list_to_docs(lvl1_categories)
                merged_cat_doc = " ".join(docs)
                self._debug(f"merged_cat_doc {merged_cat_doc}", )
            else:
                # if  self.get_selected_category_by_level(level) is not None:
                parent_category = record.get_parent_category_of_lvl(level)
                if parent_category:
                    valid_categories = parent_category.child_categories
                    docs = record.category_list_to_docs(valid_categories)
                    merged_cat_doc = " ".join(docs)
                else:
                    record.select_lvl_category(level, "cant_determined")
                    break

            generation_result = self.categorize_text_with_LLM(record.text, merged_cat_doc)
            # self._debug(f"generation_result {generation_result}")

            from indented_logger import setup_logging, smart_indent_log
            if self.debug:
                smart_indent_log(logger, generation_result, lvl=2, name="generation_result", flatten_long_strings=True)

            if generation_result.success:
                if record.validate_proposed_category_and_hierarchy(generation_result.content, level):

                    record.select_lvl_category(level, generation_result.content)
                    record.fill_rationale(level=level, value=generation_result.raw_content)
                    record.fill_refiner_output(level=level, value=generation_result.content)
                    record.categorized_by = "llm"
                else:
                    logger.debug(f"category is not valid", extra={'lvl': 3})
                    record.select_lvl_category(level, "cant_determined")

            else:
                sequential_success = False
                logger.debug(f"categorisation failed", extra={'lvl': 3})
                record.select_lvl_category(level, "cant_determined")



        if sequential_success:
            record.ready = True

        record.generate_df()
        logger.debug(f"df generated {record.df}")

    def categorize_record_with_meta_pattern(self, text, classification_patterns):

        # self.logger.debug("classify_record_with_meta_pattern()  ",  extra={'lvl': 4} )

        matched_pattern = next(
            (pattern for pattern in classification_patterns if re.search(pattern['pattern'], text)), None)

        # self.logger.debug("matched_pattern =%s ", matched_pattern, extra={'lvl': 4})

        categorization_result = CategorizationResult( success=False,
                                                      category_list=[ ],
                                                      rationale_dict={},
                                                      matched_pattern=None,
                                                      raw_llm_answer=None,
                                                      matched_keyword=None,
                                                      categorized_by=None
                                                      )
        if matched_pattern:
            categorization_result.success = True
            categorization_result.matched_pattern=matched_pattern
            categorization_result.categorized_by="metapattern"

        return categorization_result


        #if matched_pattern:
        #     # record.select_lvl_category(1, matched_pattern['lvl1'], classified_by="meta_pattern")
        #     record.select_lvl_category(1, matched_pattern['lvl1'], classified_by="meta_pattern")
        #
        #     # record.generate_merged_category_dict()
        #     # record.rationale = "metapattern"
        #     # record.refiner_output = "n"
        #     # record.generate_df()
        #     # selected_category_detail = record.category_dict
        #
        #     self.classified_by_meta_pattern_count += 1  # Increment the counter
        #     for level in range(2, self.subcategory_level + 1):
        #         lvl_key = f'lvl{level}'
        #         if lvl_key in matched_pattern:
        #             record.select_lvl_category(level, matched_pattern[lvl_key], classified_by="meta_pattern")
        #
        #     record.ready = True
        #     record.categorized_by = "metapattern"
        #     record.generate_merged_category_dict()
        #     record.rationale = ""
        #     record.refiner_output = "n"
        #     record.generate_df()
        #
        # details = {"match_exist": bool(matched_pattern), "pattern": matched_pattern}
        #
        # return details


    # def categorize_record_with_meta_pattern(self, record, classification_patterns):
    #
    #     #self.logger.debug("classify_record_with_meta_pattern()  ",  extra={'lvl': 4} )
    #
    #     matched_pattern = next(
    #         (pattern for pattern in classification_patterns if re.search(pattern['pattern'], record.text)), None)
    #
    #    # self.logger.debug("matched_pattern =%s ", matched_pattern, extra={'lvl': 4})
    #
    #     if matched_pattern:
    #         # record.select_lvl_category(1, matched_pattern['lvl1'], classified_by="meta_pattern")
    #         record.select_lvl_category(1, matched_pattern['lvl1'], classified_by="meta_pattern")
    #
    #
    #         # record.generate_merged_category_dict()
    #         # record.rationale = "metapattern"
    #         # record.refiner_output = "n"
    #         # record.generate_df()
    #         # selected_category_detail = record.category_dict
    #
    #         self.classified_by_meta_pattern_count += 1  # Increment the counter
    #         for level in range(2, self.subcategory_level + 1):
    #             lvl_key = f'lvl{level}'
    #             if lvl_key in matched_pattern:
    #                 record.select_lvl_category(level, matched_pattern[lvl_key], classified_by="meta_pattern")
    #
    #         record.ready = True
    #         record.categorized_by = "metapattern"
    #         record.generate_merged_category_dict()
    #         record.rationale = ""
    #         record.refiner_output = "n"
    #         record.generate_df()
    #
    #     details = {"match_exist": bool(matched_pattern), "pattern": matched_pattern}
    #
    #     return details

    def check_keywords_in_category(self, cat, text):
        """
        Checks if any of the category's auto_trigger_keywords are present in the text.
        Returns a tuple of (matched_keyword, cat) if a match is found, None otherwise.
        """
        for k in cat.auto_trigger_keyword:
            if k.lower() in text.lower():
                return k, cat  # Return the matched keyword and the category
        return None  # No match found

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
                parent_cat=matched_cat.parent_categories[0]
                categorization_result.success = True
                categorization_result.matched_keyword = matched_keyword
                categorization_result.categorized_by = "keyword"
                categorization_result.category_list.append(parent_cat)
                categorization_result.category_list.append(matched_cat)

                break  # Stop after finding the first match

        return categorization_result


        # self.logger.debug(" ", extra={'lvl': 6})
        # self.logger.debug("-----", extra={'lvl': 6})
        # for cat in r.available_categories:
        #     #self.logger.debug("for cat lvl =%s", i, extra={'lvl': 6})
        #     for k in cat.auto_trigger_keyword:
        #         if k.lower() in r.text.lower():
        #
        #            # self.logger.debug("MATCH, selecting category: =%s", cat.name, extra={'lvl': 11})
        #             self.classified_by_keyword_count += 1  # Increment the counter
        #             match_exist = True
        #             matched_keyword = cat.auto_trigger_keyword
        #
        #            # self.logger.debug("self.subcategory_level =%s", self.subcategory_level, extra={'lvl': 11})
        #             #self.logger.debug("max lvl =%s", r.depth, extra={'lvl': 11})
        #
        #             if cat.lvl>1:
        #                 r.select_lvl_category(cat.parent_categories[0].lvl , cat.parent_categories[0].name, classified_by="keyword")
        #             r.select_lvl_category(cat.lvl , cat.name, classified_by="keyword")
        #
        #             r.ready = True
        #
        # selected_category_detail= None
        # if match_exist:
        #     r.categorized_by= "keyword match"
        #     r.generate_merged_category_dict()
        #     r.rationale=""
        #     r.refiner_output= "n"
        #     r.generate_df()
        #     selected_category_detail= r.category_dict
        #
        #
        # details = {"match_exist": match_exist,
        #            "matched_keyword": matched_keyword,
        #            "selected_category": selected_category_detail}
        #
        # return details

    def categorize_a_record_lvl_by_lvl(self):
        pass

    @log_indent
    def categorize_text_with_LLM(self, text, classes):
       #  categorize_record_with_meta_pattern
        class ExpandedDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True
        classes_string = yaml.dump(classes, default_flow_style=False, Dumper=ExpandedDumper)

        # logger.info("inside categorize_record_with_LLM")
        generation_result=self.myllmservice.categorize_simple(text, classes_string )

        # logger.info("came back to categorization engine...(categorize_record_with_LLM)")


        return generation_result

    def _select_categories_from_pattern(self, record, pattern):
        record.select_lvl_category(1, pattern['lvl1'], classified_by="meta_pattern")
        for level in range(2, self.subcategory_level + 1):
            lvl_key = f'lvl{level}'
            if lvl_key in pattern:
                record.select_lvl_category(level, pattern[lvl_key], classified_by="meta_pattern")


    # def attempt_categorization(self, target_string, nested_categories_and_helpers):
    #
    #     generation_result = self.categorize_record_with_LLM(target_string, nested_categories_and_helpers)
    #
    #     return generation_result
    #



def main():
    from indented_logger import setup_logging, log_indent
    from indented_logger import smart_indent_log

    setup_logging(
        level=logging.DEBUG,
        include_func=True,
        truncate_messages=False,
        min_func_name_col=100,
        include_module=True,

        indent_modules=True,
        # indent_packages=indent_packages,
        # use_logger_hierarchy=True
    )

    # logger = logging.getLogger(__name__)

    # Initialize CategorizationEngine
    # categorization_engine = CategorizationEngine(logger=logger, subcategory_level=2)
    categorization_engine = CategorizationEngine( subcategory_level=2)

    # Sample data for testing
    target_string = "The company reported a significant increase in revenue this quarter."
    # nested_categories_and_helpers = [
    #
    #     {
    #         'lvl1': 'Finance',
    #         'lvl2': 'Revenue',
    #     },
    #     {
    #
    #         'lvl1': 'Business',
    #         'lvl2': 'Company Overview',
    #     }
    # ]

    categories_and_helpers = [

        {
            'category': 'Finance',
        },
        {

            'category': 'Business',
        }
    ]

    # Perform categorization
    # logger.info("Starting categorization test...")

    generation_result = categorization_engine.categorize_text_with_LLM(target_string, categories_and_helpers)

    # print(generation_result)

    # sample_dict= {"a": "aaaaa"}
    # logger_object, obj, lvl,
    smart_indent_log(logger,generation_result ,  lvl=2 ,name= "generation_result" , flatten_long_strings=True)
    # print(generation_result.content)


if __name__ == '__main__':
    main()