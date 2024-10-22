import yaml
# from prettyprint import pp

import pprint
# nested_category_manager
class MetaPatternManager:
    def __init__(self, metapattern_input='patterns.yaml'):

        self.metapattern_input = metapattern_input
        self.meta_cleaning_patterns = None
        self.meta_classification_patterns = None

        if  self.is_variable_path(metapattern_input):
            self.loaded_yaml_data = self.load_yaml(metapattern_input)
            self.loaded_yaml_data = self.loaded_yaml_data['meta_patterns']

        else:
            self.loaded_yaml_data= metapattern_input
            self.loaded_yaml_data = self.loaded_yaml_data['meta_patterns']

    def is_variable_path(self, s):
        import re
        path_regex = re.compile(
            r'^(/|\\|[a-zA-Z]:\\|\.\\|..\\|./|../)?'  # Optional start with /, \, C:\, .\, ..\, ./, or ../
            r'(?:(?:[^\\/:*?"<>|\r\n]+\\|[^\\/:*?"<>|\r\n]+/)*'  # Directory names
            r'[^\\/:*?"<>|\r\n]*)$',  # Last part of the path which can be a file
            re.IGNORECASE)
        return re.match(path_regex, s) is not None

    def print_structure(self, data, indent=0):
        """Recursively prints the structure of the given data."""
        space = '    ' * indent
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{space}{key}: (dict)")
                self.print_structure(value, indent + 1)
        elif isinstance(data, list):
            print(f"{space}(list) containing {len(data)} items")
            for item in data:
                self.print_structure(item, indent + 1)
        else:
            print(f"{space}Value: {data} (type: {type(data).__name__})")


    def bring_specific_meta_pattern(self, meta_pattern_owner, meta_pattern_name):
        """Loads specific meta patterns and sets them to instance variables."""

        # self.logger.debug("meta_pattern_owner =%s ", meta_pattern_owner, extra={'lvl': 4})
        meta_patterns=self.loaded_yaml_data[meta_pattern_owner]
        return meta_patterns[meta_pattern_name]

        # meta_patterns["cleaning_patterns"]
        # meta_patterns["classification_patterns"]


        # meta_pattern = self.load_meta_pattern(self.meta_patterns_yaml_path, meta_pattern_name)
        # self.meta_cleaning_patterns = meta_pattern["cleaning_patterns"]
        # self.meta_classification_patterns = meta_pattern["classification_patterns"]
    # def load_specific_meta_pattern(self, meta_pattern_name):
    #     """Loads specific meta patterns and sets them to instance variables."""
    #     meta_pattern = self.load_meta_pattern(self.meta_patterns_yaml_path, meta_pattern_name)
    #     self.meta_cleaning_patterns = meta_pattern["cleaning_patterns"]
    #     self.meta_classification_patterns = meta_pattern["classification_patterns"]

        # if meta_pattern_name in patterns['meta_patterns']:
        #     return patterns['meta_patterns'][meta_pattern_name]


    # def load_meta_pattern(self, file_path, meta_pattern_name):
    #     """Loads meta pattern from the given file path."""
    #     with open(file_path, 'r') as file:
    #         patterns = yaml.safe_load(file)
    #     if meta_pattern_name in patterns['meta_patterns']:
    #         return patterns['meta_patterns'][meta_pattern_name]
    #     else:
    #         raise ValueError(f"No patterns found for: {meta_pattern_name}")

    def load_yaml(self, yaml_file):
        """Loads YAML data from the specified file."""
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)

def main():
    meta_patterns_yaml_path = 'bank_patterns.yaml'
    mpm = MetaPatternManager( meta_patterns_yaml_path)
    r=mpm.bring_specific_meta_pattern("enpara", "cleaning_patterns")
    print(r)



if __name__ == "__main__":
    main()

