
import pandas as pd
import logging
from llmec.category import Category
from llmec.categorization_engine import  CategorizationEngine
from indented_logger import setup_logging, smart_indent_log
import yaml

logger = logging.getLogger(__name__)



class Record:
    def __init__(self,
                 text,
                 record_id= None,
                 max_nested_category_depth=2,
                 categories=None,
                 debug=True,
                 logger=None
                 ):

        self.logger=logger
        self.ready=False
        self.depth = None
        self.text = text
        self.cleaned_text = None
        self.keyword = None
        self.record_id = record_id
        self.associated_with = None
        self.rationale_dict = {}
        self.refiner_output_dict = {}
        self.context= None         # a record can have a context, like he is in minsk. and record says GRAY HOUSE
        self.supplementary_data= None   # a record can have a supplement_data,
        self.categorized_by = ""
        self.validated_by = None
        self.flag_for_further_inspection = False
        self.depth=None
        self.debug=debug
        self.metapatterns=None

        self.available_categories=[]
        self.categories=[]



        yaml_data=self.load_yaml(categories)
        self.create_categories_from_yaml(yaml_data["categories"])
        self.calculate_max_depth() # sets self.depth

        if isinstance(text, pd.DataFrame):
            self.text = text.get('text', pd.Series([None])).iloc[0]
            self.keyword = text.get('keyword', pd.Series([None])).iloc[0]
            self.record_id = text.get('record_id', pd.Series([None])).iloc[0]
            self.cleaned_text = text.get('cleaned_text', pd.Series([None])).iloc[0]
            self.associated_with = text.get('associated_with', pd.Series([None])).iloc[0]




        for i in range(1, self.depth + 1):
            setattr(self, f"lvl{i}", None)
            # setattr(self, f"classified_by_for_lvl{i}", None)
            setattr(self, f"lvl{i}_is_selected", False)
          #  setattr(self, f'lvl{i}_categories',getattr(self.ncm, f'lvl{i}_categories', None) if categories else None)

    def __str__(self):
        category_summary = ", ".join(
            [f"lvl{i}: {getattr(self, f'lvl{i}', 'None')}" for i in range(1, self.depth + 1)]
        )
        return (
            f"Record(\n"
            f"  Ready: {self.ready}\n"
            f"  Depth: {self.depth}\n"
            f"  Record ID: {self.record_id}\n"
            f"  Text: {self.text}\n"
            f"  Cleaned Text: {self.cleaned_text}\n"
            f"  Keyword: {self.keyword}\n"
            f"  Associated With: {self.associated_with}\n"
            f"  Rationale Dict: {self.rationale_dict}\n"
            f"  Refiner Output Dict: {self.refiner_output_dict}\n"
            f"  Selected Categories: {category_summary}\n"
            f"  Validated By: {self.validated_by}\n"
            f"  Flag for Further Inspection: {self.flag_for_further_inspection}\n"
            f")"
        )


    def _debug(self, message):
        if self.debug:
            logger.debug(message)



    def validate_proposed_category_and_hierarchy(self,value, level):


        category_list=self.filter_categories_by_lvl( level)

        for c in category_list:
           if  c.name==value:
               return True

        return False

    # def categorize_with_auto_trigger_keyword(self):
    #
    #     categorization_result= self.categorization_engine.categorize_record_with_keyword(self.available_categories, self.text)
    #     if categorization_result.success:
    #         for c in categorization_result.category_list:
    #             self.select_lvl_category(c.lvl, c.name, classified_by="keyword")

    def categorize_with_metapattern(self):

        if self.metapatterns["auto_categorization_patterns"] is not None:

            self._debug(f"self.ready {self.ready}")
            self._debug(f"inside the categorize_with_metapattern")

            classification_patterns = self.metapatterns["auto_categorization_patterns"]

            categorization_result = self.categorization_engine.categorize_record_with_meta_pattern(self.text,
                                                                                                   classification_patterns)
            self._debug(f"categorization_result: {categorization_result}")

            matched_pattern = categorization_result.matched_pattern
            self._debug(f"matched_pattern: {matched_pattern}")
            if categorization_result.success:
                self.select_lvl_category(1, matched_pattern['lvl1'], classified_by="meta_pattern")

                self.generate_merged_category_dict()
                self.rationale = "metapattern"
                self.refiner_output = "n"
                self.generate_df()
                self._debug(f"---------self.ready {self.ready}")
                selected_category_detail = self.category_dict

                # self.classified_by_meta_pattern_count += 1  # Increment the counter
                for level in range(2, self.depth + 1):
                    lvl_key = f'lvl{level}'
                    if lvl_key in matched_pattern:
                        self.select_lvl_category(level, matched_pattern[lvl_key], classified_by="meta_pattern")

    # def categorize(self, use_metapattern=False, use_keyword=False):
    #
    #     self._debug(f"categorize()")
    #
    #     sequential_success=True
    #     for level in range(1, self.depth + 1):
    #         self._debug(f"current level is {level}", )
    #         if level==1:
    #             lvl1_categories=self.filter_categories_by_lvl(level)
    #             self._debug(f"len lvl1_categories {len(lvl1_categories)}", )
    #
    #             docs = self.category_list_to_docs(lvl1_categories)
    #             merged_cat_doc = " ".join(docs)
    #             self._debug(f"merged_cat_doc {merged_cat_doc}", )
    #         else:
    #             # if  self.get_selected_category_by_level(level) is not None:
    #                 parent_category = self.get_parent_category_of_lvl(level)
    #                 if parent_category:
    #                     valid_categories = parent_category.child_categories
    #                     docs = self.category_list_to_docs(valid_categories)
    #                     merged_cat_doc = " ".join(docs)
    #                 else:
    #                     self.select_lvl_category(level, "cant_determined")
    #                     break
    #
    #
    #         generation_result = self.categorization_engine.categorize_text_with_LLM(self.text, merged_cat_doc)
    #         # self._debug(f"generation_result {generation_result}")
    #
    #
    #         smart_indent_log(logger, generation_result, lvl=2, name="generation_result", flatten_long_strings=True)
    #
    #         if generation_result.success:
    #             if self.validate_proposed_category_and_hierarchy(generation_result.content, level):
    #
    #                 self.select_lvl_category(level, generation_result.content)
    #                 self.fill_rationale(level=level, value=generation_result.raw_content)
    #                 self.fill_refiner_output(level=level, value=generation_result.content)
    #                 self.categorized_by = "llm"
    #             else:
    #                 logger.debug(f"category is not valid", extra={'lvl': 3})
    #                 self.select_lvl_category(level, "cant_determined")
    #
    #         else:
    #             sequential_success=False
    #             logger.debug(f"categorisation failed", extra={'lvl': 3})
    #             self.select_lvl_category(level, "cant_determined")
    #
    #     if sequential_success:
    #         self.ready=True


    def is_level_valid(self, level):
        if level < 1 or level > self.depth:
            return False
        return True

    def select_lvl_category(self, level, value , classified_by=None):

        if not self.is_level_valid( level):
            raise ValueError("Invalid level number")

        if level==1:
            #self.logger.debug("select_lvl_category level %s.", level, extra={'lvl': 3})

            #self.logger.debug("self.categories  %s.", self.categories, extra={'lvl': 5})
           # self.logger.debug("....value  %s.", value, extra={'lvl': 5})
            for cat in self.available_categories:
               # self.logger.debug("....cat  %s.", cat, extra={'lvl': 5})
                # self.logger.debug("cat.name  %s.", cat.name, extra={'lvl': 4})
                # self.logger.debug("value  %s.", value, extra={'lvl': 4})
                # self.logger.debug("  ", extra={'lvl': 4})
                if cat.name==value:
                    setattr(self, f'lvl{level}', cat)
                    return True
           # else:
            return False

        else:
            for cat in self.filter_categories_by_lvl( level):
               # print("cat", cat.name , "value:", value)

                if cat.name==value:
                    selected_parent_category=self.get_parent_category_of_lvl( level)
                    if cat.parent_categories[0].name == selected_parent_category.name:
                        setattr(self, f'lvl{level}', cat)
                        return True
                    else:
                        return False

            return False



    def calculate_max_depth(self):
        """Return the maximum level (depth) among the categories."""
        max_depth = 1
        for category in self.available_categories:
            if category and category.lvl > max_depth:
                max_depth = category.lvl
        self.depth=max_depth

        # return max_depth



    def category_list_to_docs(self, category_list):
        docs = []
        for cat in category_list:
            doc = cat.extract_doc()
            docs.append(doc)
        return docs

    def filter_categories_by_lvl(self, lvl):
        category_list=[]
        for c in self.available_categories:
            if c.lvl==lvl:
                category_list.append(c)

        return  category_list


    def get_selected_category_by_level(self, level):
        lvl_string = "lvl" + str(level)
        return  getattr(self, lvl_string, None)

    def get_parent_category_of_lvl(self, lvl):
        parent_lvl_string = "lvl" + str(lvl - 1)
        parent_category = getattr(self, parent_lvl_string, None)
        return parent_category

    def apply_cached_result(self, cached_record):
        # Update this record's data based on cached results
       # if cached_record.record_id is not None:
            org_record_id = self.record_id  # Store the original record ID
            #self.logger.debug("org_record_id =%s", org_record_id, extra={'lvl': 7})

            self.__dict__.update(cached_record.__dict__)  # Update the current object with cached data
            #self.logger.debug(" after df copy, record_id =%s", self.df['record_id'], extra={'lvl': 7})

            self.record_id = org_record_id  # Restore the original record ID

           # self.logger.debug("self.df['record_id']  =%s", self.df['record_id'] , extra={'lvl': 7})
            # Update the corresponding DataFrame record_id entry if df is generated
            if hasattr(self, 'df') and self.df is not None:
                self.df.at[0, 'record_id'] = org_record_id
                #self.logger.debug("after df fix, record_id = %s", self.df['record_id'], extra={'lvl': 7})

            #self.logger.debug(" after df fix, record_id =%s", self.df['record_id'], extra={'lvl': 7})


            self.categorized_by += " -cache- "


    def clone(self):
        # Create a deep copy of the record for caching purposes
        import copy
        return copy.deepcopy(self)


    # def fill_unfilled_
    # def update_category_list_based_on_determined_parent(self, lvl, parent ):
    #     _, y = self.ncm.get_categories_with_helpers(lvl, parent_filter=parent)
    #     setattr(self, f'lvl{lvl}_categories',y)

    # def all_levels_are_categorized(self):
    #     # Iterate through all levels up to self.depth
    #     for i in range(1, self.depth + 1):
    #         # Check if the attribute corresponding to this level exists and is not None
    #         if getattr(self, f'lvl{i}', None) is None:
    #             return False  # If any level is None, return False immediately
    #     return True

    def merge_dict_items_into_str(self, d):
        merged_string = " / ".join(f"{str(key)}: {value}" for key, value in d.items())
        return merged_string

    def generate_df(self):

        import pandas as pd

        if isinstance(self.text, pd.DataFrame):
            text=self.text["text"]
            if 'keyword' in self.text.columns:
                self.keyword = text['keyword'].iloc[0]
            if 'record_id' in text.columns:
                self.record_id = text['record_id'].iloc[0]
            if 'text' in text.columns:
                self.text = text['text'].iloc[0]
            if 'cleaned_text' in text.columns:
                self.cleaned_text = text['cleaned_text'].iloc[0]
        else:
            text = self.text

        data = {
            'record_id': [self.record_id],
            'record': [text],
            'category_dict': [self.category_dict],
            'rationale': [self.rationale],
            'refiner_output': [self.refiner_output],
            'associated_with': [self.associated_with],
            'categorized_by' : [self.categorized_by],

        }

        df = pd.DataFrame(data)
        self.ready=True

        self.df = df

    def generate_merged_refiner_output(self):
        r = self.merge_dict_items_into_str(self.refiner_output_dict)
        self.refiner_output = r
    def generate_merged_rationale(self):
        rationale=self.merge_dict_items_into_str(self.rationale_dict)
        self.rationale = rationale

    def process_helpers(self, category_obj, include_text_rules_for_llm=True, include_description=True,
                        remove_empty_text_rules=True):
        """
        Processes the helpers for a category, including text_rules_for_llm and description.
        Returns the helpers dictionary.
        """
        helpers_dict = {}

        # Conditionally include text_rules_for_llm
        if include_text_rules_for_llm:
            if category_obj.rules or not remove_empty_text_rules:
                helpers_dict['text_rules_for_llm'] = category_obj.rules
                print(f"Adding text rules for category: {category_obj.name}, Rules: {category_obj.rules}")

        # Conditionally include the description
        if include_description:
            helpers_dict['description'] = category_obj.desc
            print(f"Adding description for category: {category_obj.name}, Description: {category_obj.desc}")

        return helpers_dict

    def build_category_structure(self, category_obj, processed_categories, current_level,
                                 include_text_rules_for_llm=True, include_description=True, level=None,
                                 remove_empty_text_rules=True):
        """
        Recursively builds the structure for the category and its subcategories.
        """
        print(f"Processing category: {category_obj.name}, Level: {current_level}")

        if category_obj in processed_categories:
            print(f"Skipping already processed category: {category_obj.name}")
            return None  # Prevents reprocessing the same category (avoiding infinite loops)

        if level is not None and current_level > level:
            print(
                f"Skipping category due to level mismatch: {category_obj.name}, Current Level: {current_level}, Target Level: {level}")
            return None  # Skip categories that are beyond the target level

        # If we're at the target level, include this category
        if level is None or current_level == level:
            category_dict = {}
            has_content = False  # Tracks if this category has any content

            # Process helpers
            helpers_dict = self.process_helpers(category_obj, include_text_rules_for_llm, include_description,
                                                remove_empty_text_rules)

            # Only add the helpers key if it's not empty
            if helpers_dict:
                category_dict[category_obj.name] = {'helpers': helpers_dict}
                has_content = True
                print(f"Category {category_obj.name} has helpers: {helpers_dict}")
            else:
                category_dict[category_obj.name] = None

            processed_categories.add(category_obj)  # Mark this category as processed

            # Process subcategories only if we're not filtering by level or we're at the correct level
            if level is None or current_level < level:
                subcategories = [
                    subcat for subcat in self.available_categories
                    if subcat.parent_categories and subcat.parent_categories[0].name == category_obj.name
                ]

                if subcategories:  # Only include subcategories key if there are subcategories
                    subcategories_list = []
                    print(f"Processing subcategories for {category_obj.name}: {[sub.name for sub in subcategories]}")

                    for subcat in subcategories:
                        subcat_structure = self.build_category_structure(subcat, processed_categories,
                                                                         current_level + 1, include_text_rules_for_llm,
                                                                         include_description, level,
                                                                         remove_empty_text_rules)
                        if subcat_structure:  # Only append if there's a valid structure (not empty)
                            subcategories_list.append(subcat_structure)

                    if subcategories_list:
                        if category_dict[category_obj.name] is None:
                            category_dict[category_obj.name] = {}
                        category_dict[category_obj.name]['subcategories'] = subcategories_list
                        has_content = True
                        print(f"Added subcategories for {category_obj.name}: {subcategories_list}")

            # If there's no content, just set the category name as an empty value
            if not has_content:
                print(f"No content for category: {category_obj.name}, setting as empty")
                return category_obj.name

            print(f"Completed processing for category: {category_obj.name}, Result: {category_dict}")
            return category_dict

        # If we're at a higher level but still need to descend into subcategories to reach the target level
        subcategories = [
            subcat for subcat in self.available_categories
            if subcat.parent_categories and subcat.parent_categories[0].name == category_obj.name
        ]

        subcategories_list = []
        print(f"Descending into subcategories for {category_obj.name} to reach level {level}")

        for subcat in subcategories:
            subcat_structure = self.build_category_structure(subcat, processed_categories, current_level + 1,
                                                             include_text_rules_for_llm, include_description, level,
                                                             remove_empty_text_rules)
            if subcat_structure:  # Only append if there's a valid structure (not empty)
                subcategories_list.append(subcat_structure)

        if subcategories_list:
            return subcategories_list
        else:
            return None

    def extract_category_document(self, include_text_rules_for_llm=True, include_description=True, level=None,
                                  remove_empty_text_rules=True):
        """
        Extracts the category data from self.categories and converts it into a YAML-like structure.
        """
        print("Starting category extraction...")

        # Organize categories by their top-level parents
        root_categories = [cat for cat in self.available_categories if not cat.parent_categories]
        print(f"Root categories identified: {[cat.name for cat in root_categories]}")

        category_structure = []

        # Set to keep track of processed categories to avoid infinite loops
        processed_categories = set()

        for root_cat in root_categories:
            print(f"Building structure for root category: {root_cat.name}")
            root_structure = self.build_category_structure(root_cat, processed_categories, 1,
                                                           include_text_rules_for_llm, include_description, level,
                                                           remove_empty_text_rules)
            if root_structure:  # Only append if there's a valid structure (not empty)
                if isinstance(root_structure, list):
                    category_structure.extend(root_structure)  # Flatten the list
                else:
                    category_structure.append(root_structure)
                print(f"Added root structure: {root_structure}")

        # Simplify structure: Ensure proper YAML output
        simplified_category_structure = []
        for category in category_structure:
            if isinstance(category, dict):
                for key, value in category.items():
                    if value is None:
                        simplified_category_structure.append(key)
                    else:
                        simplified_category_structure.append({key: value})
            else:
                simplified_category_structure.append(category)

        print(f"Simplified category structure: {simplified_category_structure}")

        # Convert to YAML string
        yaml_string = yaml.dump({'categories': simplified_category_structure}, default_flow_style=False,
                                sort_keys=False)
        print("Category extraction completed.")
        return yaml_string


    def generate_merged_category_dict(self):
        merged_dict = {}
        for i in range(1, self.depth + 1):
            cat= getattr(self, f'lvl{i}', {})

            if cat:
                level_value = cat.name
            else:
                level_value = "cant_determined"
            level_dict= { f'lvl{i}' :level_value}
           # print("level_dict", level_dict)
            if level_dict is not None:
                merged_dict.update(level_dict)
        self.category_dict= merged_dict


    def load_yaml(self, yaml_file):
        """Loads YAML data from the specified file."""
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)


    def create_categories_from_yaml(self, yaml_data, level=1, parent=None):

       # yaml_data = yaml.safe_load(yaml_content)

        categories = []

        for category_data in yaml_data:
            for category_name, category_info in category_data.items():
                helpers = category_info.get('helpers', {})
                auto_trigger_keyword = helpers.get('keyword_identifier', [])
                text_rules_for_llm = helpers.get('text_rules_for_llm', [])
                description = helpers.get('description', "")
                subcategories = category_info.get('subcategories', [])

                # Create the Category object
                category_obj = Category(
                    name=category_name,
                    lvl=level,
                    desc=description,
                    rules=text_rules_for_llm,
                    auto_trigger_keyword=auto_trigger_keyword,
                    parent_categories=[parent] if parent else None
                )

                categories.append(category_obj)

                # Recursively create subcategories
                if subcategories:
                    subcategories_objs = self.create_categories_from_yaml(subcategories, level + 1, category_obj)
                    categories.extend(subcategories_objs)

        self.available_categories=categories
        return categories


    def fill_refiner_output(self, level, value):
        if level:
            if level < 1 or level > self.depth:
                raise ValueError("Invalid level number")
            lvl= "lvl"+str(level)

            self.refiner_output_dict[lvl]=value
        else:
            self.refiner_output_dict["general"] = value

    def fill_rationale(self, level, value, failed_attempt=False):
        msg = "(this rationale was not used  since it is deemed not valid) "
        if level:
            if level < 1 or level > self.depth:
                raise ValueError("Invalid level number")
            lvl= "lvl"+str(level)

            if failed_attempt:

                self.rationale_dict[lvl] =  msg+ value
            else:
                self.rationale_dict[lvl]=value
        else:
            if failed_attempt:
                self.rationale_dict["general"] = msg+  value
            else:

                 self.rationale_dict["general"] = value

    def select_lvl_category_as_empty(self, level, value , classified_by=None):
        if level < 1 or level > self.depth:
            raise ValueError("Invalid level number")

        setattr(self, f'lvl{level}', value)
        setattr(self, f'lvl{level}_selected', True)
        self.categorized_by = classified_by


    def validate_category_and_hierarchy(self, level, value):

        # self.logger
        self.logger.debug("validate_category_and_hierarchy ", extra={'lvl': 6})
        self.logger.debug("level =%s", level, extra={'lvl': 7})
        self.logger.debug("value =%s", value, extra={'lvl': 7})

        if level==1:
            self.logger.debug("for level 1 ", extra={'lvl': 7})
            _, list_of_categories = self.ncm.get_categories_with_helpers(level, parent_filter=None)
            #self.logger.debug("list_of_categories =%s", list_of_categories, extra={'lvl': 7})
            exists= value in list_of_categories
            self.logger.debug("category exists =%s", exists, extra={'lvl': 7})

            return exists
        else:
            self.logger.debug("for level 2> ", extra={'lvl': 7})
            parent_level=level - 1
            parent_category= getattr(self, f'lvl{parent_level}')
            self.logger.debug("parent_category =%s", parent_category, extra={'lvl': 7})

            _, list_of_categories = self.ncm.get_categories_with_helpers(level, parent_filter=parent_category)
            #self.logger.debug("list_of_categories =%s", list_of_categories, extra={'lvl': 7})

            exists = value in list_of_categories
            self.logger.debug("category exists =%s", exists, extra={'lvl': 7})
            return exists





def main():
    import logging
    import sys


    import logging

    # Configure the logger
    setup_logging(level=logging.DEBUG, include_func=True)

    categories_and_subcategories = {
        "Food": {
            "Restaurant": {
                "Fine Dining": ["Michelin Star", "Gourmet"],
                "Casual Dining": ["Family Style", "Fast Casual"]
            },
            "Grocery": {
                "Supermarket": ["Large Chain", "Local Market"],
                "Farmers Market": ["Organic", "Non-Organic"]
            }
        },
        "Transport": {
            "Bus": {
                "Local": ["City", "Suburb"],
                "Intercity": ["Short Distance", "Long Distance"]
            },
            "Train": {
                "Commuter": ["Morning", "Evening"],
                "High-Speed": ["Express", "Regular"]
            }
        }
    }



    record = Record(text="Dinner at a Michelin Star restaurant", categories='llmec/categories.yaml', debug=True)
    a=record.available_categories[0]
    print(a)

    record.categorize()

    # b = record.categories[0]

    print(". ")
    print(record.lvl1.name)

    # # y= record.extract_category_document(include_description=False, level=2)
    # # print(y)
    # print("  ")
    # print("-----")
    # b= record.filter_categories_by_lvl(2 )
    # # print(a)
    # print(b)


    # valid=record.select_lvl_category(1,'Food & Dining')
    # print(valid)
    # valid = record.select_lvl_category(2, 'Restaurants')
    # print(valid)





if __name__ == '__main__':
    main()