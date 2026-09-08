import numpy as np
import warnings
from scipy.ndimage import gaussian_filter
from AOT_biomaps.AOT_Medium._mainMedium import Medium

# Optional kwave imports
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False


class PVAMedium(Medium):
    """
    Class representing a Polyvinyl Alcohol (PVA) medium for acoustic wave propagation.
    Models a heterogeneous medium with random scattering structures.
    - The global grid remains strictly defined by user parameters (Xrange, Zrange).
    - The phantom is centered in X and starts at Z=0.
    - Background (outside the phantom) is defined by 'background_medium' ('air', 'water').
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
            raise ValueError(f"[AOT-biomaps] Unsupported background medium: {bg_medium}. Supported options are 'air' and 'water'.")

        c_map = np.full((Nx, Nz), bg_c, dtype=np.float32)
        rho_map = np.full((Nx, Nz), bg_rho, dtype=np.float32)
        alpha_coeff_map = np.zeros((Nx, Nz), dtype=np.float32)
        BonA_map = np.zeros((Nx, Nz), dtype=np.float32)

        # 5. Generate Heterogeneous PVA strictly inside the Phantom dimensions (Px, Pz)
        eta = np.random.randn(Px, Pz).astype(np.float32) * self.params.acoustic['medium']['noise_lvl']
        for _ in range(self.params.acoustic['medium']['n_phases']):
            sigma_val = np.random.uniform(*[s/dx for s in self.params.acoustic['medium']['size_structures']])
            threshold = np.random.uniform(1.2, 2.2)
            noise_field = gaussian_filter(np.random.randn(Px, Pz), sigma=sigma_val)
            noise_field /= (np.std(noise_field) + 1e-9)
            mask = 1 / (1 + np.exp(-self.params.acoustic['medium']['grad_coef'] * (noise_field - threshold)))
            zone_offset = np.random.uniform(-self.params.acoustic['medium']['c0_delta'], self.params.acoustic['medium']['c0_delta'])
            zone_grain = np.random.randn(Px, Pz).astype(np.float32) * self.params.acoustic['medium']['scattering_amplitude']
            eta = (1 - mask) * eta + mask * (zone_offset + zone_grain)

        sound_speed_pva = self.params.acoustic['medium']['c0'] * (1 + eta)
        density_pva = self.params.acoustic['medium']['density'] * (1 + eta)

        # 6. Apply PVA properties to the global maps
        c_map[x_start:x_end, z_start:z_end] = sound_speed_pva
        rho_map[x_start:x_end, z_start:z_end] = density_pva
        BonA_map[x_start:x_end, z_start:z_end] = self.params.acoustic['medium'].get('BonA', 6.0)

        # 7. Handle Absorption
        is_absorbing = self.params.acoustic['medium'].get('isAbsorbingMedium', False)
        
        if is_absorbing:
            eta_norm = (eta - np.min(eta)) / (np.max(eta) - np.min(eta) + 1e-9)
            alpha_coeff_pva = 0.4 + 0.3 * eta_norm
            alpha_coeff_map[x_start:x_end, z_start:z_end] = alpha_coeff_pva
            alpha_power = self.params.acoustic['medium'].get('alpha_power', 1.5)
            alpha_mode = 'no_dispersion'
        else:
            alpha_power = 1.5
            alpha_mode = 'no_absorption'

        # 8. Store medium properties
        self.medium_properties = {
            'sound_speed': c_map,
            'density': rho_map,
            'alpha_coeff': alpha_coeff_map,
            'alpha_power': alpha_power,
            'alpha_mode': alpha_mode,
            'BonA': BonA_map,
            'absorbing': is_absorbing,
            'sound_speed_ref': self.params.acoustic['medium']['c0']
        }

        # 9. Initialize kWave objects
        if KWAVE_AVAILABLE:
            self.kmedium = kWaveMedium(
                sound_speed=c_map,
                density=rho_map,
                sound_speed_ref=self.params.acoustic['medium']['c0'],
                alpha_coeff=alpha_coeff_map,
                alpha_power=alpha_power,
                alpha_mode=alpha_mode,
                BonA=BonA_map,
                absorbing=is_absorbing,
                stokes=False
            )

            self.kgrid = kWaveGrid([Nx, Nz], [dx, dz])
            dt = 1 / self.params.acoustic['f_AQ']
            
            nt_assigned = getattr(self, 'Nt_reshaped', self.params.general.get('Nt'))
            self.kgrid.setTime(nt_assigned, dt)
        else:
            self.kmedium = None
            self.kgrid = None
            print("[AOT-biomaps] Warning: kWave is not available. Medium properties stored in medium_properties dictionary.")

        # 10. Save variables for later use
        self.factorX = int(np.floor(self.params.general['dx'] / dx))
        self.factorZ = int(np.floor(self.params.general['dz'] / dz))
        if KWAVE_AVAILABLE and self.kgrid is not None:
            self.factorT = int(np.floor((1 / self.kgrid.dt) / self.params.acoustic['f_saving']))
        else:
            self.factorT = 1
            
        self.c_mean = np.mean(c_map[:, 0])
        self.Nx_reshaped = Nx
        self.Nz_reshaped = Nz
        self.dx_reshaped = dx
        self.dz_reshaped = dz