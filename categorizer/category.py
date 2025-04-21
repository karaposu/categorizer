#here is category.py

import yaml

import logging
from indented_logger import setup_logging, log_indent



class Category:
    def __init__(self, name, lvl, desc, rules, auto_trigger_keyword, parent_categories=None, logger=None):
        self.name = name
        self.lvl = lvl
        self.desc = desc
        self.rules = rules
        self.auto_trigger_keyword = auto_trigger_keyword
        self.parent_categories = parent_categories if parent_categories else []
        self.child_categories = []
        self.logger = logger if logger else logging.getLogger(__name__)

        # self.logger.info('This is an indented log message', extra={"lvl": 4})

        if self.parent_categories:
            for parent in self.parent_categories:
                parent.add_child_category(self, add_as_parent=False)

    def add_child_category(self, child_category, add_as_parent=True):
        if child_category not in self.child_categories:
            self.child_categories.append(child_category)

        if add_as_parent and self not in child_category.parent_categories:
            child_category.add_parent_category(self, add_as_child=False)

    def add_parent_category(self, parent_category, add_as_child=True):
        if parent_category not in self.parent_categories:
            self.parent_categories.append(parent_category)

        if add_as_child and self not in parent_category.child_categories:
            parent_category.add_child_category(self, add_as_parent=False)

    def extract_doc(self):
        import json

        # Initialize the dictionary with the mandatory field
        a = {"category": self.name}

        # Conditionally add 'rules' if it is not None and not an empty list
        if self.rules is not None and len(self.rules) > 0:
            a["rules"] = self.rules
        
        # Conditionally add 'desc' if it is not None, not an empty string, and not empty
        if self.desc is not None and self.desc.strip():
            a["desc"] = self.desc

        # Convert the dictionary to a JSON string
        b = json.dumps(a)

        return b


    def to_dict(self):

        lvl_key= "lvl" + str(self.lvl)
        t= {lvl_key: self.name}
        return  t

    def validate_bidirectionality(self):
        for parent in self.parent_categories:
            if self not in parent.child_categories:
                raise ValueError(f"Validation failed: {self.name} is not listed as a child in its parent {parent.name}.")

        for child in self.child_categories:
            if self not in child.parent_categories:
                raise ValueError(f"Validation failed: {child.name} is not listed as a parent in its child {self.name}.")

        return True

    def __repr__(self):
        return f"Category({self.name}, Level: {self.lvl}, Parent Categories: {[parent.name for parent in self.parent_categories]}, Child Categories: {[child.name for child in self.child_categories]})"


def create_categories_from_yaml(yaml_data, level=1, parent=None, logger=None):
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
                parent_categories=[parent] if parent else None,
                logger=logger
            )

            categories.append(category_obj)

            # Recursively create subcategories
            if subcategories:
                subcategories_objs = create_categories_from_yaml(subcategories, level + 1, category_obj)
                categories.extend(subcategories_objs)

    return categories


def main():
    setup_logging(
        level=logging.DEBUG,
        include_func=True,
        truncate_messages=False,
        min_func_name_col=100
    )

    logger = logging.getLogger(__name__)

    # Load the YAML file
    yaml_content = """
    categories:
      - Food & Dining:
          helpers:
            keyword_identifier: ["xyz"]
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Groceries:
                helpers:
                  keyword_identifier: ["SOSEDI", "MJET", "m-jet"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Restaurants:
                helpers:
                  keyword_identifier: ["WWW.PZZ.BY", "HOT DONER", "CHAYHONA"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Coffee:
                helpers:
                  keyword_identifier: ["KOFEYNYA", "GRAY HOUSE"]
                  text_rules_for_llm:
                    - "If record contains words like coffea or KAFE, it must be categorized as Coffee."
                  description: ""
                subcategories: []
            - Takeout:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Utilities:
          helpers:
            keyword_identifier: ["vvc"]
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Electricity and Water and Gas:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Internet and Mobile:
                helpers:
                  keyword_identifier: ["Türk Telekom Mobil"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Accommodation:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Accommodation:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Incoming P2P Transfers:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Incoming Money:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Outgoing P2P Transfers:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Outgoing Money:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm:
                    - "Must be towards a person, not to a company name."
                  description: ""
                subcategories: []
    
      - Cash Withdrawal:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Cash Withdrawal:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Cash Deposit:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Cash Deposit:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Transportation:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Fuel:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Taxi:
                helpers:
                  keyword_identifier: ["YANDEX.GO"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Travel Tickets:
                helpers:
                  keyword_identifier: ["HAVAIST"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Public Transportation:
                helpers:
                  keyword_identifier: ["IGA", "ISPARK"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Vehicle Maintenance:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Car Payments:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Healthcare:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Medical Bills:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Health Insurance:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Medications:
                helpers:
                  keyword_identifier: ["ECZANE", "APTEKA"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Retail Purchases:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Clothes:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Technology Items:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm:
                    - "If bill is from APPLE and cost is less than $40, it should be classified as an online subscription."
                    - "If bill is from Getir and cost is more than $50, it should be classified as retail purchases."
                  description: ""
                subcategories: []
            - Other:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Personal Care:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Personal Grooming:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Fitness:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Leisure and Activities in Real Life:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Movies:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Concerts:
                helpers:
                  keyword_identifier: []
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    
      - Online Subscriptions & Services:
          helpers:
            keyword_identifier: []
            text_rules_for_llm: []
            description: ""
          subcategories:
            - Streaming & Digital Subscriptions:
                helpers:
                  keyword_identifier: ["NETFLIX", "HULU", "AMAZON PRIME", "youtube+", "SPOTIFY", "APPLE MUSIC", "GOOGLE PLAY"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
            - Cloud Server Payments:
                helpers:
                  keyword_identifier: ["AWS", "AMAZON WEB SERVICES", "GOOGLE CLOUD", "GCP", "GOOGLE CLOUD PLATFORM", "DIGITALOCEAN"]
                  text_rules_for_llm: []
                  description: ""
                subcategories: []
    """
    yaml_data = yaml.safe_load(yaml_content)
    categories_list = create_categories_from_yaml(yaml_data['categories'], logger=logger)

    print(categories_list[0])

    print(categories_list[0].child_categories[0])

    print(categories_list[0].child_categories[0].parent_categories[0])


if __name__ == '__main__':
    main()
