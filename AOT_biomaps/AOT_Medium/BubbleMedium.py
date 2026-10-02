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
    Class representing a medium with random air bubbles for acoustic wave
    propagation.
    - The SIMULATION grid is defined by the acoustic parameters
      (dx_sim/dz_sim, Nx_sim/Nz_sim), finer than (or equal to) the saved grid.
    - The phantom is centered in X and starts at Z=0.
    - Background (outside the phantom) is water or air.
    - Respects Nyquist and optimizes VRAM (float32, in-place calculations).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_medium(self):
        """
        Generate a bubbly medium on the SIMULATION grid.
        """
        # Simulation grid (fine resolution)
        dx = float(self.params.acoustic.get('dx_sim', self.params.general['dx']))
        dz = float(self.params.acoustic.get('dz_sim', self.params.general['dz']))
        Nx = int(self.params.acoustic.get('Nx_sim', self.params.general['Nx']))
        Nz = int(self.params.acoustic.get('Nz_sim', self.params.general['Nz']))

        width = self.params.acoustic['medium'].get('width', self.params.general['Xrange'][1] - self.params.general['Xrange'][0])
        height = self.params.acoustic['medium'].get('height', self.params.general['Zrange'][1] - self.params.general['Zrange'][0])

        Px = int(np.round(width / dx))
        Pz = int(np.round(height / dz))

        # Security: ensure phantom doesn't exceed the simulation grid
        Px = min(Px, Nx)
        Pz = min(Pz, Nz)

        # Positioning the phantom (centered in X, top in Z)
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

        # Fill the phantom background (water/gel)
        phantom_c = self.params.acoustic['medium']['c0']
        phantom_rho = self.params.acoustic['medium']['density']
        phantom_alpha = self.params.acoustic['medium'].get('alpha_coeff', 0.5)
        phantom_BonA = self.params.acoustic['medium'].get('BonA', 6.0)

        c_map[x_start:x_end, z_start:z_end] = phantom_c
        rho_map[x_start:x_end, z_start:z_end] = phantom_rho
        alpha_coeff_map[x_start:x_end, z_start:z_end] = phantom_alpha
        BonA_map[x_start:x_end, z_start:z_end] = phantom_BonA

        # ------------------------------------------------------------------
        # Random air bubbles, strictly inside the phantom.
        # FIX: the two axes of the ogrid are now correctly assigned:
        # rows (axis 0) = X, cols (axis 1) = Z. In the previous version,
        # the X center was subtracted from the Z axis (and vice versa),
        # which misplaced the bubbles whenever Px != Pz and could even
        # push them outside the mask (silently lost bubbles).
        # ------------------------------------------------------------------
        bubble_mask = np.zeros((Px, Pz), dtype=np.float32)
        n_bubbles = self.params.acoustic['medium'].get('n_bubbles', 50)
        min_bubble_radius = self.params.acoustic['medium'].get('min_bubble_radius', 2)
        max_bubble_radius = self.params.acoustic['medium'].get('max_bubble_radius', 5)

        X_ax, Z_ax = np.ogrid[:Px, :Pz]

        for _ in range(n_bubbles):
            radius = np.random.randint(min_bubble_radius, max_bubble_radius)
            # Ensure bubbles don't cross phantom boundaries
            bc_x = np.random.randint(radius, Px - radius) if Px > 2 * radius else Px // 2
            bc_z = np.random.randint(radius, Pz - radius) if Pz > 2 * radius else Pz // 2

            dist_from_center_sq = (X_ax - bc_x) ** 2 + (Z_ax - bc_z) ** 2
            bubble_mask[dist_from_center_sq <= radius * radius] = 1.0

        # Apply bubbles (air: c=343, rho=1.2) to the phantom region
        c_map[x_start:x_end, z_start:z_end] = np.where(bubble_mask == 1.0, 343.0, c_map[x_start:x_end, z_start:z_end])
        rho_map[x_start:x_end, z_start:z_end] = np.where(bubble_mask == 1.0, 1.2, rho_map[x_start:x_end, z_start:z_end])
        alpha_coeff_map[x_start:x_end, z_start:z_end] = np.where(bubble_mask == 1.0, 0.0, alpha_coeff_map[x_start:x_end, z_start:z_end])
        BonA_map[x_start:x_end, z_start:z_end] = np.where(bubble_mask == 1.0, 0.0, BonA_map[x_start:x_end, z_start:z_end])

        # Absorption flag: from the parameter schema (consistent with the
        # other media; previously hardcoded to True)
        is_absorbing = self.params.acoustic['medium'].get('isAbsorbingMedium', True)
        if is_absorbing:
            alpha_power = self.params.acoustic['medium'].get('alpha_power', 1.5)
            alpha_mode = None
        else:
            alpha_power = 1.5
            alpha_mode = 'no_absorption'

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

        # Save variables for later use (SIMULATION grid attributes)
        self.c_mean = np.mean(c_map[:, 0])
        self.Nx_reshaped = Nx
        self.Nz_reshaped = Nz
        self.dx_reshaped = dx
        self.dz_reshaped = dz