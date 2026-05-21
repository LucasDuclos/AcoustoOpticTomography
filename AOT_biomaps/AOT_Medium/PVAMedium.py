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
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_medium(self):
        dx = self.params.general['dx']
        if dx >= self.params.acoustic['probe']['element_width']:
            dx = self.params.acoustic['probe']['element_width'] / 2
        width = self.params.acoustic['medium'].get('width', self.params.general['Xrange'][1] - self.params.general['Xrange'][0])
        height = self.params.acoustic['medium'].get('height', self.params.general['Zrange'][1] - self.params.general['Zrange'][0])
        pva_nx = int(np.round(width / dx))
        pva_nz = int(np.round(height / dx))

        air_margin = 20 if self.params.acoustic['medium']['isAirReflection'] else 0
        Nx, Nz = pva_nx + 2 * air_margin, pva_nz

        c_map = np.full((Nx, Nz), 343.0 if air_margin else self.params.acoustic['medium']['c0'], dtype=np.float32)
        rho_map = np.full((Nx, Nz), 1.2 if air_margin else self.params.acoustic['medium']['density'], dtype=np.float32)
        BonA_map = np.zeros((Nx, Nz), dtype=np.float32)

        x_start = air_margin
        x_end = x_start + pva_nx

        eta = np.random.randn(pva_nx, pva_nz).astype(np.float32) * self.params.acoustic['medium']['noise_lvl']
        for _ in range(self.params.acoustic['medium']['n_phases']):
            sigma_val = np.random.uniform(*[s/self.params.general['dx'] for s in self.params.acoustic['medium']['size_structures']])
            threshold = np.random.uniform(1.2, 2.2)
            noise_field = gaussian_filter(np.random.randn(pva_nx, pva_nz), sigma=sigma_val)
            noise_field /= (np.std(noise_field) + 1e-9)
            mask = 1 / (1 + np.exp(-self.params.acoustic['medium']['grad_coef'] * (noise_field - threshold)))
            zone_offset = np.random.uniform(-self.params.acoustic['medium']['c0_delta'], self.params.acoustic['medium']['c0_delta'])
            zone_grain = np.random.randn(pva_nx, pva_nz).astype(np.float32) * self.params.acoustic['medium']['scattering_amplitude']
            eta = (1 - mask) * eta + mask * (zone_offset + zone_grain)

        sound_speed_pva = self.params.acoustic['medium']['c0'] * (1 + eta)
        density_pva = self.params.acoustic['medium']['density'] * (1 + eta)

        c_map[x_start:x_end, :] = sound_speed_pva
        rho_map[x_start:x_end, :] = density_pva

        is_absorbing = self.params.acoustic['medium']['isAbsorbingMedium']
        if is_absorbing:
            eta_norm = (eta - np.min(eta)) / (np.max(eta) - np.min(eta) + 1e-9)
            alpha_coeff_pva = 0.4 + 0.3 * eta_norm
            alpha_coeff_map = np.zeros((Nx, Nz), dtype=np.float32)
            alpha_coeff_map[x_start:x_end, :] = alpha_coeff_pva
            alpha_mode = 'no_dispersion'
        else:
            alpha_coeff_map = np.zeros((Nx, Nz), dtype=np.float32)
            alpha_mode = 'no_absorption'

        if not air_margin:
            BonA_map[:, :] = self.params.acoustic['medium'].get('BonA', 6.0)
        else:
            BonA_map[x_start:x_end, :] = self.params.acoustic['medium'].get('BonA', 6.0)

        self.medium_properties = {
            'sound_speed': c_map,
            'density': rho_map,
            'alpha_coeff': alpha_coeff_map,
            'alpha_power': self.params.acoustic['medium']['alpha_power'],
            'alpha_mode': alpha_mode,
            'BonA': BonA_map,
            'absorbing': is_absorbing,
            'sound_speed_ref': self.params.acoustic['medium']['c0']
        }

        if KWAVE_AVAILABLE:
            self.kmedium = kWaveMedium(
                sound_speed=c_map,
                density=rho_map,
                sound_speed_ref=self.params.acoustic['medium']['c0'],
                alpha_coeff=alpha_coeff_map,
                alpha_power=self.params.acoustic['medium']['alpha_power'],
                alpha_mode=alpha_mode,
                BonA=BonA_map,
                absorbing=is_absorbing,
                stokes=False
            )

            self.kgrid = kWaveGrid([Nx, Nz], [dx, dx])
            dt = 1/(self.params.acoustic['f_AQ'])
            self.kgrid.setTime(self.Nt_reshaped, dt)
        else:
            self.kmedium = None
            self.kgrid = None
            warnings.warn("kWave is not available. Medium properties stored in medium_properties dictionary.", UserWarning)

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
