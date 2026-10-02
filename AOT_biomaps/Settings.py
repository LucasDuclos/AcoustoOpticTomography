import json
import numpy as np


class Params:
    def __init__(self, path):
        config = self._init_params(path)
        print(f"[AOT-biomaps] Configuration loaded from {path}")
        self.general = config.get('general', {})
        self.acoustic = config.get('acoustic', {})
        self.optic = config.get('optic', {})
        self.reconstruction = config.get('reconstruction', {})

        # ------------------------------------------------------------------
        # FINAL (saved / reconstruction) grid: general.dx, general.dz
        # This is the resolution of the fields AFTER the band-limited
        # resampling performed in generate_acoustic_field_KWAVE_2D.
        # ------------------------------------------------------------------
        self.general['Nx'] = int(np.round((self.general['Xrange'][1] - self.general['Xrange'][0]) / self.general['dx']))
        self.general['Ny'] = int(np.round((self.general['Yrange'][1] - self.general['Yrange'][0]) / self.general['dy'])) if self.general['Yrange'] is not None else 1
        self.general['Nz'] = int(np.round((self.general['Zrange'][1] - self.general['Zrange'][0]) / self.general['dz']))

        # ------------------------------------------------------------------
        # SIMULATION grid: acoustic.dx_sim, acoustic.dz_sim
        # Backward compatible: defaults to the general grid when absent
        # ------------------------------------------------------------------
        self.acoustic.setdefault('dx_sim', self.general['dx'])
        self.acoustic.setdefault('dy_sim', self.general['dy'])
        self.acoustic.setdefault('dz_sim', self.general['dz'])
        self.acoustic['Nx_sim'] = int(np.round((self.general['Xrange'][1] - self.general['Xrange'][0]) / self.acoustic['dx_sim']))
        self.acoustic['Ny_sim'] = int(np.round((self.general['Yrange'][1] - self.general['Yrange'][0]) / self.acoustic['dy_sim'])) if self.general['Yrange'] is not None else 1
        self.acoustic['Nz_sim'] = int(np.round((self.general['Zrange'][1] - self.general['Zrange'][0]) / self.acoustic['dz_sim']))

        # ------------------------------------------------------------------
        # Temporal sampling rates
        #   f_AQ : simulation sampling rate (None here if 'AUTO'; the Medium resolves it from the CFL condition)
        #   ft   : TARGET sampling rate of the SAVED fields. The temporal decimation is band-limited (spectral), performed on GPU in generate_acoustic_field_KWAVE_2D. The EFFECTIVE step is stored at generation time in field.last_decimation.
        # ------------------------------------------------------------------
        self.acoustic['f_AQ'] = None if self.acoustic['f_AQ'] == 'AUTO' else int(float(self.acoustic['f_AQ']))
        self.acoustic['f_US'] = int(float(self.acoustic['f_US']))
        self.general['ft'] = int(float(self.general['ft'])) if self.general.get('ft') is not None else None
        self.general['Nt'] = self.general.get('Nt')

        # ------------------------------------------------------------------
        # Consistency checks
        # ------------------------------------------------------------------
        if self.general['dx'] < self.acoustic['dx_sim'] - 1e-12 or \
        self.general['dz'] < self.acoustic['dz_sim'] - 1e-12:
            raise ValueError(
                "[AOT-biomaps] The saved grid step (general.dx/dz) must be >= the "
                "simulation grid step (acoustic.dx_sim/dz_sim): one cannot save at "
                "a finer resolution than the simulation."
            )
        if self.general['ft'] is not None and self.acoustic['f_AQ'] is not None:
            if self.general['ft'] > self.acoustic['f_AQ']:
                raise ValueError(
                    "[AOT-biomaps] ft (saved sampling rate) cannot exceed "
                    "f_AQ (simulation sampling rate)."
                )

        self.acoustic['medium']['size_structures'] = [float(s) for s in self.acoustic['medium']['size_structures']]

        if self.acoustic['medium']['width'] > self.general['Xrange'][1] - self.general['Xrange'][0]:
            raise ValueError("[AOT-biomaps] The medium width must be smaller than the X range to ensure it fills the grid.")
        if self.acoustic['medium']['height'] > self.general['Zrange'][1] - self.general['Zrange'][0]:
            raise ValueError("[AOT-biomaps] The medium height must be smaller than the Z range to ensure it fills the grid.")

        if self.acoustic['medium']['background_medium'].lower() not in ['air', 'water']:
            raise ValueError("[AOT-biomaps] Unsupported background medium: {}. Supported options are 'air' and 'water'.".format(self.acoustic['medium']['background_medium']))
        if self.acoustic['medium']['background_medium'].lower() == 'air':
            x_range_width = self.general['Xrange'][1] - self.general['Xrange'][0]
            # 40 SIMULATION pixels of air margin (the sim grid is what 'sees' the wave)
            required_width = self.acoustic['medium']['width'] + 40 * self.acoustic['dx_sim']
            if required_width > x_range_width + 1e-9:
                excess_mm = (required_width - x_range_width) * 1e3
                raise ValueError(
                    f"[AOT-biomaps] Configuration conflict: With 'background_medium' set to 'air', the medium width + 40 pixels of air margin "
                    f"exceeds the global Xrange by {excess_mm:.2f} mm. "
                    f"Please either increase Xrange or decrease the medium width."
                )
        
    def __repr__(self):
        return (f"[AOT-biomaps] Params(general={self.general}, acoustic={self.acoustic}, optic={self.optic}, "
                f"reconstruction={self.reconstruction})")

    def _init_params(self, path):
        if not path.endswith('.json'):
            raise ValueError("[AOT-biomaps] The configuration file must be a JSON file with a .json extension.")
        try:
            with open(path, 'r') as file:
                config = json.load(file)
                if config is None:
                    raise ValueError("[AOT-biomaps] The configuration file is empty or not valid JSON.")
                if 'Parameters' in config:
                    config = config['Parameters']
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"[AOT-biomaps] The file {path} does not exist.")
        except json.JSONDecodeError as e:
            raise ValueError(f"[AOT-biomaps] Error parsing JSON file: {e}")

    def show_parameters(self):
        config = {
            'general': self.general,
            'acoustic': self.acoustic,
            'optic': self.optic,
            'reconstruction': self.reconstruction
        }
        self._print_config(config)

    def save_to_json(self, path):
        """
        Save the current configuration to a JSON file. The structure of the JSON file will be as follows:
        {
        "Parameters": {
            "general": {...},
            "acoustic": {...},
            "optic": {...},
            "reconstruction": {...}
        }
        }
        """
        if not path.endswith('.json'):
            raise ValueError("[AOT-biomaps] The output file must have a .json extension.")
        config = {
            "Parameters": {
                "general": self.general,
                "acoustic": self.acoustic,
                "optic": self.optic,
                "reconstruction": self.reconstruction
            }
        }
        try:
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=4, ensure_ascii=False)
            print(f"[AOT-biomaps] Configuration saved to {path}")
        except Exception as e:
            raise IOError(f"[AOT-biomaps] Error occurred while saving the file: {e}")

    def _print_config(self, config, indent=0):
        border = "+" + "-" * 100 + "+"
        print(border)
        print(f"|{'[AOT-biomaps] Configuration Loaded '.center(100)}|")
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
