import json
import numpy as np


class Params:
    def __init__(self, path):
        config = self._init_params(path)
        print(f"Configuration loaded from {path}")
        self.general = config.get('general', {})
        self.acoustic = config.get('acoustic', {})
        self.optic = config.get('optic', {})
        self.reconstruction = config.get('reconstruction', {})
        self.general['Nx'] = int(np.round((self.general['Xrange'][1] - self.general['Xrange'][0]) / self.general['dx']))
        self.general['Ny'] = int(np.round((self.general['Yrange'][1] - self.general['Yrange'][0]) / self.general['dy'])) if self.general['Yrange'] is not None else 1
        self.general['Nz'] = int(np.round((self.general['Zrange'][1] - self.general['Zrange'][0]) / self.general['dz']))
        self.general['Nt'] = int((self.general['Nt']) * int(float(self.acoustic['f_AQ'])) / int(float(self.acoustic['f_saving']))) if 'Nt' in self.general else None
        if self.acoustic['f_AQ'] == 'AUTO':
            self.acoustic['f_AQ'] = None
        self.acoustic['f_AQ'] = int(float(self.acoustic['f_AQ']))
        self.acoustic['f_saving'] = int(float(self.acoustic['f_saving']))
        self.acoustic['f_US'] = int(float(self.acoustic['f_US']))
        self.acoustic['medium']['size_structures'] = [float(s) for s in self.acoustic['medium']['size_structures']]

    def __repr__(self):
        return (f"Params(general={self.general}, acoustic={self.acoustic}, optic={self.optic}, "
                f"reconstruction={self.reconstruction})")

    def _init_params(self, path):
        if not path.endswith('.json'):
            raise ValueError("The configuration file must be a JSON file with a .json extension.")
        try:
            with open(path, 'r') as file:
                config = json.load(file)
                if config is None:
                    raise ValueError("The configuration file is empty or not valid JSON.")
                if 'Parameters' in config:
                    config = config['Parameters']
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {path} does not exist.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file: {e}")

    def show_parameters(self):
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
