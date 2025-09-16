import yaml

class Params:
    def __init__(self, path):
        config = self._initParams(path)
        print(f"Configuration loaded from {path}")
        self.general = config.get('general', {})
        self.acoustic = config.get('acoustic', {})
        self.optic = config.get('optic', {})
        self.reconstruction = config.get('reconstruction', {})

    def __repr__(self):
        return (f"Params(general={self.general}, acoustic={self.acoustic}, optic={self.optic}, "
                f"reconstruction={self.reconstruction})")

    def _initParams(self, path):
        if not path.endswith('.yaml'):
            raise ValueError("The configuration file must be a YAML file with a .yaml extension.")
        try:
            with open(path, 'r') as file:
                config = yaml.safe_load(file)
                if config is None:
                    raise ValueError("The configuration file is empty or not valid YAML.")
                if 'Parameters' in config:
                    config = config['Parameters']
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {path} does not exist.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def show_Parameters(self):
        config = {
            'general': self.general,
            'acoustic': self.acoustic,
            'optic': self.optic,
            'reconstruction': self.reconstruction
        }
        self._print_config(config)

    def _print_config(self, config, indent=0):
        border = "+" + "-" * 100 + "+"
        print(border)
        print(f"|{' Configuration Loaded '.center(100)}|")
        print(border)
        categories = {
            'General': config.get('general', {}),
            'Acoustic': config.get('acoustic', {}),
            'Optic': config.get('optic', {}),
            'Reconstruction': config.get('reconstruction', {})
        }
        for category, params in categories.items():
            print("|" + category.center(100) + "|")
            print(border)
            self._print_params(params, indent + 2)
            print(border)

    def _print_params(self, params, indent):
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    print(f"|{' ' * indent}{key}:")
                    self._print_params(value, indent + 2)
                else:
                    print(f"|{' ' * indent}{key}: {value}")
        elif isinstance(params, list):
            for item in params:
                if isinstance(item, (dict, list)):
                    self._print_params(item, indent)
                else:
                    print(f"|{' ' * indent}- {item}")
