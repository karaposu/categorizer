import yaml


class MetaPatternManager:
    def __init__(self, metapattern_input='patterns.yaml'):

        self.metapattern_input = metapattern_input
       
        self.meta_classification_patterns = None
        
        if  self.looks_like_path(metapattern_input):
            self.loaded_yaml_data = self.load_yaml(metapattern_input)
            self.loaded_yaml_data = self.loaded_yaml_data['meta_patterns']

        else:
            self.loaded_yaml_data= metapattern_input
            self.loaded_yaml_data = self.loaded_yaml_data['meta_patterns']

    def looks_like_path(self, s):
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

