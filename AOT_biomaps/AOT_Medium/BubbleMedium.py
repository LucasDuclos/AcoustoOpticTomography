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
    - The global grid remains strictly defined by user parameters (Xrange, Zrange).
    - The phantom is centered in X and starts at Z=0.
    - Background (outside the phantom) is filled with water or air.
    - Respects Nyquist and optimizes VRAM (float32, in-place calculations).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_medium(self):
        dx = self.params.general['dx']
        dz = self.params.general['dz']
        
        Nx = int(self.params.general['Nx'])
        Nz = int(self.params.general['Nz'])

        width = self.params.acoustic['medium'].get('width', self.params.general['Xrange'][1] - self.params.general['Xrange'][0])
        height = self.params.acoustic['medium'].get('height', self.params.general['Zrange'][1] - self.params.general['Zrange'][0])
        
        Px = int(np.round(width / dx))
        Pz = int(np.round(height / dz))
        
        # Security: ensure phantom doesn't exceed global grid
        Px = min(Px, Nx)
        Pz = min(Pz, Nz)

        # Positioning the Phantom (Centered in X, Top in Z)
        x_start = (Nx - Px) // 2
        x_end = x_start + Px
        z_start = 0
        z_end = Pz

        bg_medium = self.params.acoustic['medium'].get('background_medium', 'water').lower()
        
        if bg_medium == 'air':
            bg_c = 343.0
            bg_rho = 1.2
        elif bg_medium == 'water':
            bg_c = self.params.acoustic['medium']['c0']
            bg_rho = self.params.acoustic['medium']['density']
        else:
            raise ValueError(f"Unsupported background medium: {bg_medium}. Supported options are 'air' and 'water'.")

        c_map = np.full((Nx, Nz), bg_c, dtype=np.float32)
        rho_map = np.full((Nx, Nz), bg_rho, dtype=np.float32)
        alpha_coeff_map = np.zeros((Nx, Nz), dtype=np.float32)
        BonA_map = np.zeros((Nx, Nz), dtype=np.float32)

        # Fill the Phantom Background (Water/Gel)
        phantom_c = self.params.acoustic['medium']['c0']
        phantom_rho = self.params.acoustic['medium']['density']
        phantom_alpha = self.params.acoustic['medium'].get('alpha_coeff', 0.5)
        phantom_BonA = self.params.acoustic['medium'].get('BonA', 6.0)

        c_map[x_start:x_end, z_start:z_end] = phantom_c
        rho_map[x_start:x_end, z_start:z_end] = phantom_rho
        alpha_coeff_map[x_start:x_end, z_start:z_end] = phantom_alpha
        BonA_map[x_start:x_end, z_start:z_end] = phantom_BonA


        bubble_mask = np.zeros((Px, Pz), dtype=np.float32)
        n_bubbles = self.params.acoustic['medium'].get('n_bubbles', 50)
        min_bubble_radius = self.params.acoustic['medium'].get('min_bubble_radius', 2)
        max_bubble_radius = self.params.acoustic['medium'].get('max_bubble_radius', 5)

        for _ in range(n_bubbles):
            radius = np.random.randint(min_bubble_radius, max_bubble_radius)
            # Ensure bubbles don't cross phantom boundaries
            bc_x = np.random.randint(radius, Px - radius) if Px > 2*radius else Px//2
            bc_z = np.random.randint(radius, Pz - radius) if Pz > 2*radius else Pz//2
            
            Y, X = np.ogrid[:Px, :Pz]
            dist_from_center = np.sqrt((X - bc_x)**2 + (Y - bc_z)**2)
            bubble_mask[dist_from_center <= radius] = 1.0

        # Apply Bubbles (Air: c=343, rho=1.2) to the phantom region
        c_map[x_start:x_end, z_start:z_end] = np.where(bubble_mask == 1.0, 343.0, c_map[x_start:x_end, z_start:z_end])
        rho_map[x_start:x_end, z_start:z_end] = np.where(bubble_mask == 1.0, 1.2, rho_map[x_start:x_end, z_start:z_end])
        alpha_coeff_map[x_start:x_end, z_start:z_end] = np.where(bubble_mask == 1.0, 0.0, alpha_coeff_map[x_start:x_end, z_start:z_end])
        BonA_map[x_start:x_end, z_start:z_end] = np.where(bubble_mask == 1.0, 0.0, BonA_map[x_start:x_end, z_start:z_end])

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

            self.kgrid = kWaveGrid([Nx, Nz], [dx, dz])
            dt = 1 / self.params.acoustic['f_AQ']
            self.kgrid.setTime(self.params.general['Nt'], dt)
        else:
            self.kmedium = None
            self.kgrid = None
            warnings.warn("kWave is not available. Medium properties stored in medium_properties dictionary.", UserWarning)

        self.factorX = int(np.ceil(self.params.general['dx'] / dx))
        self.factorZ = int(np.ceil(self.params.general['dz'] / dz))
        if KWAVE_AVAILABLE and self.kgrid is not None:
            self.factorT = int(np.ceil((1/self.kgrid.dt) / self.params.acoustic['f_saving']))
        else:
            self.factorT = 1
            
        self.c_mean = np.mean(c_map[:, 0])
        self.Nx_reshaped = Nx
        self.Nz_reshaped = Nz
        self.dx_reshaped = dx
        self.dz_reshaped = dz