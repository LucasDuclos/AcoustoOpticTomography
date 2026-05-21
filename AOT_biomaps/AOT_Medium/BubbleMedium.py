import numpy as np
import warnings
from AOT_biomaps.AOT_Medium._mainMedium import Medium

# Optional kwave imports
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False


class BubbleMedium(Medium):
    """
    Class representing a medium with random air bubbles for acoustic wave propagation.
    - Air bubbles are randomly distributed in a background medium (e.g., water or gel).
    - Optional air margin around the medium.
    - Respects Nyquist and optimizes VRAM (float32, in-place calculations).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_medium(self):
        """
        Generate a medium with random air bubbles.
        - If isAirReflection=True: air surrounds the medium (20-pixel margins).
        - Otherwise: the medium occupies the entire grid.
        """
        dx = self.params.general['dx']
        if dx >= self.params.acoustic['probe']['element_width']:
            dx = self.params.acoustic['probe']['element_width'] / 2

        # Physical dimensions to pixels
        width = self.params.acoustic['medium'].get('width',
                self.params.general['Xrange'][1] - self.params.general['Xrange'][0])
        height = self.params.acoustic['medium'].get('height',
                self.params.general['Zrange'][1] - self.params.general['Zrange'][0])
        nx = int(np.round(width / dx))
        nz = int(np.round(height / dx))

        # Add air margins if needed
        air_margin = 20 if self.params.acoustic['medium']['isAirReflection'] else 0
        Nx, Nz = nx + 2 * air_margin, nz

        c_map = np.full((Nx, Nz), 343.0 if air_margin else self.params.acoustic['medium']['c0'], dtype=np.float32)
        rho_map = np.full((Nx, Nz), 1.2 if air_margin else self.params.acoustic['medium']['density'], dtype=np.float32)
        alpha_coeff_map = np.zeros((Nx, Nz), dtype=np.float32)
        BonA_map = np.zeros((Nx, Nz), dtype=np.float32)

        x_start = air_margin
        x_end = x_start + nx

        # Background map (water/gel)
        c_background = self.params.acoustic['medium']['c0']
        rho_background = self.params.acoustic['medium']['density']

        # Mask for air bubbles
        bubble_mask = np.zeros((nx, nz), dtype=np.float32)
        n_bubbles = self.params.acoustic['medium'].get('n_bubbles', 50)
        min_bubble_radius = self.params.acoustic['medium'].get('min_bubble_radius', 2)
        max_bubble_radius = self.params.acoustic['medium'].get('max_bubble_radius', 5)

        for _ in range(n_bubbles):
            radius = np.random.randint(min_bubble_radius, max_bubble_radius)
            x_center = np.random.randint(radius, nx - radius)
            z_center = np.random.randint(radius, nz - radius)
            Y, X = np.ogrid[:nx, :nz]
            dist_from_center = np.sqrt((X - x_center)**2 + (Y - z_center)**2)
            bubble_mask[dist_from_center <= radius] = 1.0

        # Air bubbles: c=343 m/s, rho=1.2 kg/m^3
        c_map[x_start:x_end, :] = np.where(
            bubble_mask == 1.0,
            343.0,
            c_background
        )
        rho_map[x_start:x_end, :] = np.where(
            bubble_mask == 1.0,
            1.2,
            rho_background
        )
        alpha_coeff_map[x_start:x_end, :] = np.where(
            bubble_mask == 1.0,
            0.0,
            self.params.acoustic['medium'].get('alpha_coeff', 0.5)
        )
        BonA_map[x_start:x_end, :] = np.where(
            bubble_mask == 1.0,
            0.0,
            self.params.acoustic['medium'].get('BonA', 6.0)
        )

        # Store medium properties
        self.medium_properties = {
            'sound_speed': c_map,
            'density': rho_map,
            'alpha_coeff': alpha_coeff_map,
            'BonA': BonA_map,
            'sound_speed_ref': self.params.acoustic['medium']['c0'],
            'alpha_power': self.params.acoustic['medium']['alpha_power']
        }

        if KWAVE_AVAILABLE:
            self.kmedium = kWaveMedium(
                sound_speed=c_map,
                density=rho_map,
                sound_speed_ref=self.params.acoustic['medium']['c0'],
                alpha_coeff=alpha_coeff_map,
                alpha_power=self.params.acoustic['medium']['alpha_power'],
                BonA=BonA_map,
                absorbing=True,
                stokes=False
            )

            self.kgrid = kWaveGrid([Nx, Nz], [dx, dx])
            dt = 1/(self.params.acoustic['f_AQ'])
            self.kgrid.setTime(self.params.general['Nt'], dt)
        else:
            self.kmedium = None
            self.kgrid = None
            warnings.warn("kWave is not available. Medium properties stored in medium_properties dictionary.", UserWarning)

        # Save variables for later use
        self.factorX = int(np.ceil(self.params.general['dx'] / dx))
        self.factorZ = self.factorX
        if KWAVE_AVAILABLE and self.kgrid is not None:
            self.factorT = int(np.ceil((1/self.kgrid.dt) / (self.params.acoustic['f_saving'])))
        else:
            self.factorT = 1
        self.c_mean = np.mean(c_map[:, 0])
        self.Nx_reshaped = Nx
        self.Nz_reshaped = Nz
        self.dx_reshaped = dx
