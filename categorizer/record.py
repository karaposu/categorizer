import logging
import yaml
from categorizer.category import Category  # Adjust the import path as needed
from indented_logger import setup_logging
import pandas as pd

class Record:
    def __init__(
        self,
        text,
        record_id=None,
        keyword=None,
        cleaned_text=None,
        associated_with=None,
        categories=None,
        debug=True,
        logger=None,
    ):
        self.text = text
        self.record_id = record_id
        self.keyword = keyword
        self.cleaned_text = cleaned_text
        self.associated_with = associated_with
        self.categories = categories
        self.debug = debug
        self.logger = logger if logger else logging.getLogger(__name__)

        # Initialize additional attributes
        self.ready = False
        self.depth = None
        self.rationale_dict = {}
        self.refiner_output_dict = {}
        self.context = None
        self.supplementary_data = None
        self.categorized_by = ""
        self.validated_by = None
        self.flag_for_further_inspection = False
        self.metapatterns = None
        self.available_categories = []
        self.selected_categories = []

        # Load categories from YAML if provided
        if categories:
            yaml_data = self.load_yaml(categories)
            self.create_categories_from_yaml(yaml_data["categories"])
            self.calculate_max_depth()  # sets self.depth

            # Initialize level attributes based on depth
            if self.depth:
                for i in range(1, self.depth + 1):
                    setattr(self, f"lvl{i}", None)
                    setattr(self, f"lvl{i}_is_selected", False)

    @classmethod
    def from_string(cls, text, record_id=None, categories=None, **kwargs):
        return cls(text=text, record_id=record_id, categories=categories, **kwargs)

    @classmethod
    def from_dataframe(cls, df_row, categories=None, **kwargs):
        return cls(
            text=df_row.get('text', ''),
            record_id=df_row.get('record_id'),
            keyword=df_row.get('keyword'),
            cleaned_text=df_row.get('cleaned_text'),
            associated_with=df_row.get('associated_with'),
            categories=categories,
            **kwargs
        )

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
            # f"  Selected Categories: {category_summary}\n"
            f"  Validated By: {self.validated_by}\n"
            f"  Flag for Further Inspection: {self.flag_for_further_inspection}\n"
            f")"
        )

    def _debug(self, message):
        if self.debug:
            self.logger.debug(message)

    def validate_proposed_category_and_hierarchy(self, value, level):
        category_list = self.filter_categories_by_lvl(level)
        for c in category_list:
            if c.name == value:
                return True
        return False

   

    
    def is_level_valid(self, level):
        return 1 <= level <= self.depth

    def select_lvl_category(self, level, value, classified_by=None):
        if not self.is_level_valid(level):
            raise ValueError("Invalid level number")

        if level == 1:
            for cat in self.available_categories:
                if cat.name == value:
                    setattr(self, f'lvl{level}', cat)
                    return True
            return False
        else:
            for cat in self.filter_categories_by_lvl(level):
                if cat.name == value:
                    selected_parent_category = self.get_parent_category_of_lvl(level)
                    if cat.parent_categories[0].name == selected_parent_category.name:
                        setattr(self, f'lvl{level}', cat)
                        return True
            return False

    def calculate_max_depth(self):
        max_depth = 1
        for category in self.available_categories:
            if category and category.lvl > max_depth:
                max_depth = category.lvl
        self.depth = max_depth

    def category_list_to_docs(self, category_list):
        docs = []
        for cat in category_list:
            doc = cat.extract_doc()
            docs.append(doc)
        return docs

    def filter_categories_by_lvl(self, lvl):
        return [c for c in self.available_categories if c.lvl == lvl]

    def get_selected_category_by_level(self, level):
        return getattr(self, f'lvl{level}', None)

    def get_parent_category_of_lvl(self, lvl):
        return getattr(self, f'lvl{lvl - 1}', None)

    def apply_cached_result(self, cached_record):
        org_record_id = self.record_id
        self.__dict__.update(cached_record.__dict__)
        self.record_id = org_record_id
        self.categorized_by += " -cache- "

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def merge_dict_items_into_str(self, d):
        return " / ".join(f"{str(key)}: {value}" for key, value in d.items())

    def to_dict(self):
        data = {
            'ready': self.ready,
            'record_id': self.record_id,
            'record': self.text,
            'category_dict': getattr(self, 'category_dict', None),
            'rationale': getattr(self, 'rationale', None),
            'refiner_output': getattr(self, 'refiner_output', None),
            'associated_with': self.associated_with,
            'categorized_by': self.categorized_by,
            # Include other fields as needed
        }

        # Dynamically include level attributes based on self.depth
        if self.depth:
            for level in range(1, self.depth + 1):
                level_attr = getattr(self, f'lvl{level}', None)
                # Include the name of the category if it's set, else None
                data[f'lvl{level}'] = level_attr.name if level_attr else None

        return data

    def generate_merged_refiner_output(self):
        self.refiner_output = self.merge_dict_items_into_str(self.refiner_output_dict)

    def generate_merged_rationale(self):
        self.rationale = self.merge_dict_items_into_str(self.rationale_dict)

    def process_helpers(
        self,
        category_obj,
        include_text_rules_for_llm=True,
        include_description=True,
        remove_empty_text_rules=True,
    ):
        helpers_dict = {}
        if include_text_rules_for_llm:
            if category_obj.rules or not remove_empty_text_rules:
                helpers_dict['text_rules_for_llm'] = category_obj.rules
               # self._debug(f"Adding text rules for category: {category_obj.name}, Rules: {category_obj.rules}")

        if include_description:
            helpers_dict['description'] = category_obj.desc
          #  self._debug(f"Adding description for category: {category_obj.name}, Description: {category_obj.desc}")

        return helpers_dict

    def build_category_structure(
        self,
        category_obj,
        processed_categories,
        current_level,
        include_text_rules_for_llm=True,
        include_description=True,
        level=None,
        remove_empty_text_rules=True,
    ):
      #  self._debug(f"Processing category: {category_obj.name}, Level: {current_level}")

        if category_obj in processed_categories:
          #  self._debug(f"Skipping already processed category: {category_obj.name}")
            return None

        if level is not None and current_level > level:
           # self._debug(  f"Skipping category due to level mismatch: {category_obj.name}, Current Level: {current_level}, Target Level: {level}"  )
            return None

        if level is None or current_level == level:
            category_dict = {}
            has_content = False

            helpers_dict = self.process_helpers(
                category_obj, include_text_rules_for_llm, include_description, remove_empty_text_rules
            )

            if helpers_dict:
                category_dict[category_obj.name] = {'helpers': helpers_dict}
                has_content = True
              #  self._debug(f"Category {category_obj.name} has helpers: {helpers_dict}")
            else:
                category_dict[category_obj.name] = None

            processed_categories.add(category_obj)

            if level is None or current_level < level:
                subcategories = [
                    subcat for subcat in self.available_categories
                    if subcat.parent_categories and subcat.parent_categories[0].name == category_obj.name
                ]

                if subcategories:
                    subcategories_list = []
                  #  self._debug(   f"Processing subcategories for {category_obj.name}: {[sub.name for sub in subcategories]}"  )

                    for subcat in subcategories:
                        subcat_structure = self.build_category_structure(
                            subcat,
                            processed_categories,
                            current_level + 1,
                            include_text_rules_for_llm,
                            include_description,
                            level,
                            remove_empty_text_rules,
                        )
                        if subcat_structure:
                            subcategories_list.append(subcat_structure)

                    if subcategories_list:
                        if category_dict[category_obj.name] is None:
                            category_dict[category_obj.name] = {}
                        category_dict[category_obj.name]['subcategories'] = subcategories_list
                        has_content = True
                     #   self._debug(f"Added subcategories for {category_obj.name}: {subcategories_list}")

            if not has_content:
               # self._debug(f"No content for category: {category_obj.name}, setting as empty")
                return category_obj.name

           # self._debug(f"Completed processing for category: {category_obj.name}, Result: {category_dict}")
            return category_dict

        subcategories = [
            subcat for subcat in self.available_categories
            if subcat.parent_categories and subcat.parent_categories[0].name == category_obj.name
        ]

        subcategories_list = []
      #  self._debug(f"Descending into subcategories for {category_obj.name} to reach level {level}")

        for subcat in subcategories:
            subcat_structure = self.build_category_structure(
                subcat,
                processed_categories,
                current_level + 1,
                include_text_rules_for_llm,
                include_description,
                level,
                remove_empty_text_rules,
            )
            if subcat_structure:
                subcategories_list.append(subcat_structure)

        if subcategories_list:
            return subcategories_list
        else:
            return None

    def extract_category_document(
        self,
        include_text_rules_for_llm=True,
        include_description=True,
        level=None,
        remove_empty_text_rules=True,
    ):
       # self._debug("Starting category extraction...")

        root_categories = [cat for cat in self.available_categories if not cat.parent_categories]
       # self._debug(f"Root categories identified: {[cat.name for cat in root_categories]}")

        category_structure = []
        processed_categories = set()

        for root_cat in root_categories:
           # self._debug(f"Building structure for root category: {root_cat.name}")
            root_structure = self.build_category_structure(
                root_cat,
                processed_categories,
                1,
                include_text_rules_for_llm,
                include_description,
                level,
                remove_empty_text_rules,
            )
            if root_structure:
                if isinstance(root_structure, list):
                    category_structure.extend(root_structure)
                else:
                    category_structure.append(root_structure)
              #  self._debug(f"Added root structure: {root_structure}")

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

       # self._debug(f"Simplified category structure: {simplified_category_structure}")

        yaml_string = yaml.dump(
            {'categories': simplified_category_structure},
            default_flow_style=False,
            sort_keys=False,
        )
       # self._debug("Category extraction completed.")
        return yaml_string

    def generate_merged_category_dict(self):
        merged_dict = {}
        for i in range(1, self.depth + 1):
            cat = getattr(self, f'lvl{i}', {})
            level_value = cat.name if cat else "cant_determined"
            level_dict = {f'lvl{i}': level_value}
            if level_dict is not None:
                merged_dict.update(level_dict)
        self.category_dict = merged_dict

    def load_yaml(self, yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)

    def create_categories_from_yaml(self, yaml_data, level=1, parent=None):
        categories = []

        for category_data in yaml_data:
            for category_name, category_info in category_data.items():
                helpers = category_info.get('helpers', {})
                auto_trigger_keyword = helpers.get('keyword_identifier', [])
                text_rules_for_llm = helpers.get('text_rules_for_llm', [])
                description = helpers.get('description', "")
                subcategories = category_info.get('subcategories', [])

                category_obj = Category(
                    name=category_name,
                    lvl=level,
                    desc=description,
                    rules=text_rules_for_llm,
                    auto_trigger_keyword=auto_trigger_keyword,
                    parent_categories=[parent] if parent else None
                )

                categories.append(category_obj)

                if subcategories:
                    subcategories_objs = self.create_categories_from_yaml(subcategories, level + 1, category_obj)
                    categories.extend(subcategories_objs)

        self.available_categories = categories
        return categories

    def fill_refiner_output(self, level, value):
        if level:
            if not self.is_level_valid(level):
                raise ValueError("Invalid level number")
            lvl = f'lvl{level}'
            self.refiner_output_dict[lvl] = value
        else:
            self.refiner_output_dict["general"] = value

    def fill_rationale(self, level, value, failed_attempt=False):
        msg = "(this rationale was not used since it is deemed not valid) "
        if level:
            if not self.is_level_valid(level):
                raise ValueError("Invalid level number")
            lvl = f'lvl{level}'
            self.rationale_dict[lvl] = msg + value if failed_attempt else value
        else:
            self.rationale_dict["general"] = msg + value if failed_attempt else value

    def select_lvl_category_as_empty(self, level, value, classified_by=None):
        if not self.is_level_valid(level):
            raise ValueError("Invalid level number")
        setattr(self, f'lvl{level}', value)
        setattr(self, f'lvl{level}_selected', True)
        self.categorized_by = classified_by

    def validate_category_and_hierarchy(self, level, value):
        self.logger.debug("Validating category and hierarchy")
        if level == 1:
            exists = any(cat.name == value for cat in self.available_categories if cat.lvl == 1)
            self.logger.debug(f"Category exists at level 1: {exists}")
            return exists
        else:
            parent_level = level - 1
            parent_category = getattr(self, f'lvl{parent_level}')
            if not parent_category:
                return False
            exists = any(
                cat.name == value and cat.parent_categories[0].name == parent_category.name
                for cat in self.available_categories
                if cat.lvl == level
            )
            self.logger.debug(f"Category exists at level {level}: {exists}")
            return exists


def main():
    # Configure the logger
    setup_logging(level=logging.DEBUG, include_func=True)

    # Test Record.from_string
    print("Testing Record.from_string:")
    record_from_string = Record.from_string(
        text="Dinner at a Michelin Star restaurant",
        record_id=1,
        categories='categorizer/categories.yaml',  # Update the path to your categories.yaml
        debug=True
    )

    # Print the Record to see if it initializes correctly
    print("\nRecord after initialization from string:")
    print(record_from_string)

    # Optionally, test methods of the Record class
    category_doc = record_from_string.extract_category_document()
    # print("\nExtracted category document from string record:")
    # print(category_doc)

    # Test Record.from_dataframe
    # print("\nTesting Record.from_dataframe:")

    # Create a sample DataFrame
    data = {
        'text': ["Lunch at a local organic farmers market"],
        'record_id': [2],
        'keyword': ["organic"],
        'cleaned_text': ["Lunch at farmers market"],
        'associated_with': ["Food"]
    }
    df = pd.DataFrame(data)

    # Assuming the DataFrame has at least one row
    df_row = df.iloc[0]

    record_from_dataframe = Record.from_dataframe(
        df_row,
        categories='categorizer/categories.yaml',  # Update the path to your categories.yaml
        debug=True
    )

    # Print the Record to see if it initializes correctly
    print("\nRecord after initialization from DataFrame:")
    print(record_from_dataframe)

    # # Optionally, test methods of the Record class
    # category_doc_df = record_from_dataframe.extract_category_document()
    # print("\nExtracted category document from DataFrame record:")
    # print(category_doc_df)


if __name__ == '__main__':
    main()

