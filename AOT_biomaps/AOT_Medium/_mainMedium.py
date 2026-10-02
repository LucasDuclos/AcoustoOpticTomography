from abc import ABC, abstractmethod
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Optional kwave imports
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False

# Bump when the on-disk medium format changes
MEDIUM_FORMAT_VERSION = 2


class Medium(ABC):
    
    def __init__(self, params):
        """
        Base constructor.

        Builds the SIMULATION kgrid from the parameter schema
        (acoustic.dx_sim/dz_sim, Nx_sim/Nz_sim), resolves the simulation
        sampling rate f_AQ from the CFL condition when AUTO, and derives
        the SAVE TARGETS consumed by generate_acoustic_field_KWAVE_2D:
            dx_save/dz_save : general.dx/dz  (final grid step)
            dt_save         : 1 / general.ft (final time step); None when
                              ft is unset -> no temporal decimation.

        factorT is always 1: the source lives at the simulation rate, and
        temporal decimation is post-simulation (spectral, driven by
        dt_save). Nx_reshaped/dx_reshaped/... are the SIMULATION grid
        attributes, set by the subclass generate_medium().
        """
        self.params = params
        self.c_mean = None
        self.Nx_reshaped = None
        self.Nz_reshaped = None
        self.dx_reshaped = None
        self.dz_reshaped = None
        self.medium_properties = None

        # ------------------------------------------------------------------
        # Save targets: consumed as DEFAULTS by generate_acoustic_field_KWAVE_2D.
        # dx_save/dz_save/dt_save are the resampling goals of the final fields.
        # ------------------------------------------------------------------
        self.dx_save = float(self.params.general['dx'])
        self.dz_save = float(self.params.general['dz'])
        self.dt_save = (1.0 / float(self.params.general['ft'])) if self.params.general.get('ft') else None

        if KWAVE_AVAILABLE:
            # --------------------------------------------------------------
            # The kgrid is built on the SIMULATION grid (dx_sim / Nx_sim), which is finer than (or equal to) the saved grid (general.dx).
            # --------------------------------------------------------------
            Nx_sim = int(self.params.acoustic['Nx_sim'])
            Nz_sim = int(self.params.acoustic['Nz_sim'])
            dx_sim = float(self.params.acoustic['dx_sim'])
            dz_sim = float(self.params.acoustic['dz_sim'])

            self.kgrid = kWaveGrid([Nx_sim, Nz_sim], [dx_sim, dz_sim])

            # Simulation sampling rate: resolved from the CFL condition if AUTO
            if self.params.acoustic['f_AQ'] in (None, "AUTO"):
                self.kgrid.makeTime(self.params.acoustic['medium']['c0'])
                self.params.acoustic['f_AQ'] = int(1 / self.kgrid.dt)

            # Simulation step count: from params if provided, else from geometry
            if self.params.general['Nt'] in (None, "None"):
                Lx = Nx_sim * dx_sim
                Lz = self.params.general['Zrange'][1] - self.params.general['Zrange'][0]
                theta = np.radians(20)
                distance_max = (Lx * np.sin(theta)) + (Lz * np.cos(theta))
                f_aq = float(self.params.acoustic['f_AQ'])
                c0 = float(self.params.acoustic['medium']['c0'])
                Nt_strict = distance_max * f_aq / c0
                margin = 1.05
                Nt = int(np.ceil(Nt_strict * margin))
                self.params.general['Nt'] = Nt
            else:
                Nt = int(self.params.general['Nt'])

            self.kgrid.setTime(Nt, 1 / float(self.params.acoustic['f_AQ']))

            if self.dt_save is not None and self.dt_save < self.kgrid.dt:
                print(f"[AOT-biomaps] Warning: dt_save ({self.dt_save*1e9:.1f} ns) is finer than the simulation step ({self.kgrid.dt*1e9:.1f} ns). Temporal decimation disabled.")
                self.dt_save = None

        else:
            self.kgrid = None
            self.Nt_reshaped = self.params.general.get('Nt', 400)
            print("[AOT-biomaps] Warning: kWave is not available. Using default values for grid parameters.")

    @abstractmethod
    def generate_medium(self):
        """
        Abstract method to generate the medium properties.
        This method should be implemented by subclasses.
        """
        pass

    def save_medium(self, folderPath, fileName="medium"):
        """
        Save the medium state to a .npy file (pickled dict).
        Universally handles any subclass (HomogeneousMedium, PVAMedium, ...).

        An explicit metadata block records the medium identity and geometry:
            __format_version__ : on-disk format version (checked at load:
                                 files written by a NEWER library are refused)
            __class_name__      : the subclass that created the file
                                 (validated at load: load into the SAME
                                 subclass; there is no auto-dispatch factory)
            __absorbing__       : whether the medium is absorbing (restored
                                 from the file at load, never from params)
            __grid__            : simulation grid geometry
                                 (Nx, Nz, dx, dz, Nt, dt; checked at load
                                 against the current parameter schema)
            __kmedium_data__    : raw kWave medium tensors

        Note: self.params and the save targets (dx_save, dz_save, dt_save)
        are intentionally NOT pickled. The caller's Params (fresh JSON) is
        the source of truth at load time; resurrecting stale values would
        silently revert schema changes.
        """
        if os.path.splitext(fileName)[1]:
            raise ValueError("[AOT-biomaps] The fileName should not contain an extension; .npy will be added automatically.")

        os.makedirs(folderPath, exist_ok=True)
        filePath = os.path.join(folderPath, fileName + '.npy')

        if os.path.isdir(filePath):
            raise IsADirectoryError(f"[AOT-biomaps] Cannot save medium: {filePath} is a directory.")

        # ------------------------------------------------------------------
        # kmedium decomposition (raw arrays, kwave-free pickle)
        # ------------------------------------------------------------------
        kmedium_data = {}
        if getattr(self, 'kmedium', None) is not None:
            props = ['sound_speed', 'density', 'alpha_coeff', 'alpha_mode',
                     'BonA', 'alpha_power', 'absorbing', 'sound_speed_ref', 'stokes']
            for p in props:
                kmedium_data[p] = getattr(self.kmedium, p, None)

        # ------------------------------------------------------------------
        # Absorbing flag: kmedium first, then medium_properties, then params
        # ------------------------------------------------------------------
        if kmedium_data.get('absorbing') is not None:
            is_absorbing = bool(kmedium_data['absorbing'])
        elif getattr(self, 'medium_properties', None) is not None and 'absorbing' in self.medium_properties:
            is_absorbing = bool(self.medium_properties['absorbing'])
        else:
            is_absorbing = bool(self.params.acoustic['medium'].get('isAbsorbingMedium', False))

        # ------------------------------------------------------------------
        # Simulation grid geometry snapshot
        # ------------------------------------------------------------------
        kgrid = getattr(self, 'kgrid', None)
        grid_meta = {
            'Nx': int(getattr(self, 'Nx_reshaped', self.params.acoustic.get('Nx_sim', self.params.general.get('Nx')))),
            'Nz': int(getattr(self, 'Nz_reshaped', self.params.acoustic.get('Nz_sim', self.params.general.get('Nz')))),
            'dx': float(getattr(self, 'dx_reshaped', self.params.acoustic.get('dx_sim', self.params.general.get('dx')))),
            'dz': float(getattr(self, 'dz_reshaped', self.params.acoustic.get('dz_sim', self.params.general.get('dz')))),
            'Nt': int(getattr(self, 'Nt_reshaped', None) or self.params.general.get('Nt') or 1),
            'dt': float(kgrid.dt) if (kgrid is not None and getattr(kgrid, 'dt', None)) else None,
        }

        # ------------------------------------------------------------------
        # Assemble the state (params, kgrid and kmedium excluded)
        # ------------------------------------------------------------------
        state_to_save = {
            '__format_version__': MEDIUM_FORMAT_VERSION,
            '__class_name__': self.__class__.__name__,
            '__absorbing__': is_absorbing,
            '__grid__': grid_meta,
        }

        for key, value in self.__dict__.items():
            if key in ('kgrid', 'kmedium', 'params', 'dx_save', 'dz_save', 'dt_save'):
                continue
            state_to_save[key] = value

        state_to_save['__kmedium_data__'] = kmedium_data

        np.save(filePath, state_to_save, allow_pickle=True)

    def load_medium(self, folderPath, fileName="medium",
                    allow_class_mismatch=False, allow_grid_mismatch=False):
        """
        Load a medium saved by save_medium().

        Restored FROM THE FILE:
            - the ABSORBING state (__absorbing__, never taken from params),
            - the medium state (maps, c_mean, factors, ...).

        Validated against the file (ERROR by default):
            - the medium TYPE: load into the subclass that created the file
              (the file's __class_name__ is checked; there is no
              auto-dispatch factory). Remedy on mismatch: instantiate the
              named subclass yourself, or allow_class_mismatch=True.
            - the SIMULATION GRID: the saved geometry is compared to the
              current schema (dx_sim/Nx_sim/dz_sim/Nz_sim, and the physical
              extents implied by Xrange/Zrange), with a full diff in the
              error message. Rationale: the simulation would stay
              internally consistent, so the failure would be SILENT and
              DELAYED (headers FOV, optical maps, recon geometry assume
              the CURRENT schema). Remedy: regenerate the medium, or
              allow_grid_mismatch=True.
            - the FORMAT VERSION: files written by a newer library format
              are refused (a loud refusal beats a partial silent load).

        Design decisions:
            - The caller's `self.params` (fresh JSON) is the source of truth
              for the parameter schema. A pickled Params inside an old file
              is ignored.
            - The save targets (dx_save, dz_save, dt_save) are re-derived from
              the CURRENT schema after loading: they are schema-driven, not
              medium state.
            - A time-axis difference (Nt/dt) is legitimate (the medium maps
              do not depend on the temporal sampling): reported as an Info
              line, never an error.
        """
        if os.path.splitext(fileName)[1]:
            raise ValueError("[AOT-biomaps] The fileName should not contain an extension; .npy will be added automatically.")

        filePath = os.path.join(folderPath, fileName + '.npy')
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"[AOT-biomaps] The file {filePath} does not exist.")

        loaded_state = np.load(filePath, allow_pickle=True).item()

        if not isinstance(loaded_state, dict) or '__class_name__' not in loaded_state:
            raise ValueError(f"[AOT-biomaps] {filePath} is not a medium save file (missing __class_name__).")

        # ------------------------------------------------------------------
        # 1. Pop the metadata block (+ format version check)
        # ------------------------------------------------------------------
        saved_class = loaded_state.pop('__class_name__')

        # The file is typed: refuse files written by a NEWER library format (a partial silent load is worse than a loud refusal).
        file_version = loaded_state.pop('__format_version__', None)
        if file_version is not None and int(file_version) > MEDIUM_FORMAT_VERSION:
            raise ValueError(f"[AOT-biomaps] The medium file '{filePath}' was written by a NEWER format (v{file_version}) than this library supports (v{MEDIUM_FORMAT_VERSION}). Update AOT-biomaps before loading it.")
        if file_version is None:
            print("[AOT-biomaps] Info: legacy medium file (no format version tag).")

        is_absorbing_saved = loaded_state.pop('__absorbing__', None)
        grid_meta = loaded_state.pop('__grid__', None) or {}
        kmedium_data = loaded_state.pop('__kmedium_data__', {})

        # ------------------------------------------------------------------
        # 2. Type check: the file IS typed -> enforce it
        # ------------------------------------------------------------------
        if saved_class not in (self.__class__.__name__, 'Medium'):
            msg = (f"[AOT-biomaps] Medium type mismatch: the file was saved by '{saved_class}' but you are loading into '{self.__class__.__name__}'.")
            if allow_class_mismatch:
                print(f"{msg} Proceeding anyway (allow_class_mismatch=True).")
            else:
                raise TypeError(msg + f" Instantiate '{saved_class}' directly with the current params and load into it, or pass allow_class_mismatch=True to force this load.")

        # ------------------------------------------------------------------
        # 3. Restore the state (NEVER resurrect a stale pickled Params)
        # ------------------------------------------------------------------
        loaded_state.pop('params', None)
        self.__dict__.update(loaded_state)

        if getattr(self, 'medium_properties', None) is not None:
            if 'sound_speed' in self.medium_properties:
                self.c_map = self.medium_properties['sound_speed']
            if 'density' in self.medium_properties:
                self.rho_map = self.medium_properties['density']
        
        # Re-derive the SAVE TARGETS from the CURRENT schema. They are schema-driven (not medium state): restoring them from the pickle would silently revert JSON changes of dx/dz/ft.
        self.dx_save = float(self.params.general['dx'])
        self.dz_save = float(self.params.general['dz'])
        self.dt_save = (1.0 / float(self.params.general['ft'])) if self.params.general.get('ft') else None

        # ------------------------------------------------------------------
        # 4. Absorbing state: FROM THE SAVE, not from the caller
        # ------------------------------------------------------------------
        if is_absorbing_saved is not None:
            is_absorbing = bool(is_absorbing_saved)
        elif kmedium_data.get('absorbing') is not None:
            is_absorbing = bool(kmedium_data['absorbing'])
        elif getattr(self, 'medium_properties', None) and 'absorbing' in self.medium_properties:
            is_absorbing = bool(self.medium_properties['absorbing'])
        else:
            # Legacy file (format v1, no __absorbing__ tag): fall back to params
            is_absorbing = bool(self.params.acoustic['medium'].get('isAbsorbingMedium', False))
            print("[AOT-biomaps] Warning: legacy medium file without __absorbing__ tag; "
                  "absorbing state taken from the current params.")

        # Sync the parameter schema so downstream code sees the restored state
        self.params.acoustic['medium']['isAbsorbingMedium'] = is_absorbing

        # ------------------------------------------------------------------
        # 5. Rebuild kgrid / kmedium on the SAVED simulation grid
        # ------------------------------------------------------------------
        if KWAVE_AVAILABLE and hasattr(self, 'params'):
            Nx = int(getattr(self, 'Nx_reshaped', grid_meta.get('Nx', self.params.acoustic.get('Nx_sim', self.params.general.get('Nx', 1)))))
            Nz = int(getattr(self, 'Nz_reshaped', grid_meta.get('Nz', self.params.acoustic.get('Nz_sim', self.params.general.get('Nz', 1)))))
            dx = float(getattr(self, 'dx_reshaped', grid_meta.get('dx', self.params.acoustic.get('dx_sim', self.params.general.get('dx', 1)))))
            dz = float(getattr(self, 'dz_reshaped', grid_meta.get('dz', self.params.acoustic.get('dz_sim', self.params.general.get('dz', 1)))))

            # --------------------------------------------------------------
            # Grid consistency check: SAVED medium vs CURRENT schema.
            # ERROR by default: the failure would otherwise be silent and delayed (headers / optical maps / recon geometry assume the current schema).
            # --------------------------------------------------------------
            Nx_sim = int(self.params.acoustic.get('Nx_sim', self.params.general['Nx']))
            Nz_sim = int(self.params.acoustic.get('Nz_sim', self.params.general['Nz']))
            dx_sim = float(self.params.acoustic.get('dx_sim', self.params.general['dx']))
            dz_sim = float(self.params.acoustic.get('dz_sim', self.params.general['dz']))

            diffs = []
            if Nx != Nx_sim:
                diffs.append(f"Nx:      saved {Nx} vs params {Nx_sim}")
            if Nz != Nz_sim:
                diffs.append(f"Nz:      saved {Nz} vs params {Nz_sim}")
            if abs(dx - dx_sim) > 1e-12:
                diffs.append(f"dx:      saved {dx*1e6:.1f} um vs params {dx_sim*1e6:.1f} um")
            if abs(dz - dz_sim) > 1e-12:
                diffs.append(f"dz:      saved {dz*1e6:.1f} um vs params {dz_sim*1e6:.1f} um")

            # Physical extents (the quantity that actually poisons the downstream geometry when it differs)
            extent_x_saved = Nx * dx
            extent_x_params = self.params.general['Xrange'][1] - self.params.general['Xrange'][0]
            extent_z_saved = Nz * dz
            extent_z_params = self.params.general['Zrange'][1] - self.params.general['Zrange'][0]
            if abs(extent_x_saved - extent_x_params) > 1e-9:
                diffs.append(f"X extent: saved {extent_x_saved*1e3:.2f} mm vs params {extent_x_params*1e3:.2f} mm "
                             f"(headers/optics/recon assume the params extent)")
            if abs(extent_z_saved - extent_z_params) > 1e-9:
                diffs.append(f"Z extent: saved {extent_z_saved*1e3:.2f} mm vs params {extent_z_params*1e3:.2f} mm "
                             f"(headers/optics/recon assume the params extent)")

            if diffs:
                msg = ("[AOT-biomaps] Medium/params GRID MISMATCH — the saved medium "
                       "was generated with a different simulation geometry:\n  " +
                       "\n  ".join(diffs) +
                       "\n  The simulation itself would stay internally consistent, but every "
                       "downstream geometric consumer (field headers FOV, optical absorber "
                       "positions, S-matrix and reconstruction geometry) assumes the CURRENT "
                       "params geometry and would be silently wrong.\n"
                       "  Fix: regenerate the medium with the current params (recommended), "
                       "or pass allow_grid_mismatch=True if this is intentional.")
                if allow_grid_mismatch:
                    print(f"{msg}\n  -> Proceeding anyway (allow_grid_mismatch=True): "
                          f"the SAVED grid is used so medium and kgrid stay consistent.")
                else:
                    raise ValueError(msg)

            self.kgrid = kWaveGrid([Nx, Nz], [dx, dz])

            Nt = int(getattr(self, 'Nt_reshaped', None) or grid_meta.get('Nt') or self.params.general.get('Nt') or 1)
            dt_sim = grid_meta.get('dt')
            if dt_sim is None and self.params.acoustic.get('f_AQ'):
                dt_sim = 1.0 / float(self.params.acoustic['f_AQ'])
            if Nt > 1 and dt_sim is not None:
                self.kgrid.setTime(Nt, float(dt_sim))

            if self.dt_save is not None and self.dt_save < self.kgrid.dt:
                print(f"[AOT-biomaps] Warning: dt_save ({self.dt_save*1e9:.1f} ns) is finer than the simulation step ({self.kgrid.dt*1e9:.1f} ns). Temporal decimation disabled.")
                self.dt_save = None

            # ------------------------------------------------------------------
            # kmedium: absorbing state restored from the save, not from params
            # ------------------------------------------------------------------
            kmedium_kwargs = {k: v for k, v in kmedium_data.items() if v is not None}
            if is_absorbing:
                kmedium_kwargs['absorbing'] = True
            else:
                kmedium_kwargs['absorbing'] = False
                kmedium_kwargs['alpha_mode'] = 'no_absorption'
                kmedium_kwargs['alpha_coeff'] = np.zeros((Nx, Nz), dtype=np.float32)

            if 'sound_speed' in kmedium_kwargs:
                self.kmedium = kWaveMedium(**kmedium_kwargs)
            else:
                self.kmedium = None
        else:
            self.kgrid = None
            self.kmedium = None

        if is_absorbing:
            print("[AOT-biomaps] Info: loaded medium is ABSORBING (restored from the save file).")
        else:
            print("[AOT-biomaps] Info: loaded medium is NON-ABSORBING (restored from the save file).")
            
    def plot_medium_properties(self, figsize=(12, 5),vmin_speed=None, vmax_speed=None, vmin_density=None, vmax_density=None):
        if not KWAVE_AVAILABLE:
            print("[AOT-biomaps] Warning: kWave is not available. Cannot plot medium properties.")
            return
        
        if getattr(self, 'kmedium', None) is None:
            raise ValueError("[AOT-biomaps] Medium properties are not available. Please generate or load the medium first.")
            
        if vmin_speed is None:
            vmin_speed = np.min(self.kmedium.sound_speed)
        if vmax_speed is None:
            vmax_speed = np.max(self.kmedium.sound_speed)
        if vmin_density is None:
            vmin_density = np.min(self.kmedium.density)
        if vmax_density is None:
            vmax_density = np.max(self.kmedium.density)

        extent = [self.params.general['Xrange'][0]*1e3, self.params.general['Xrange'][1]*1e3, 
                  self.params.general['Zrange'][1]*1e3, self.params.general['Zrange'][0]*1e3]
        
        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.imshow(self.kmedium.sound_speed.T, vmin=vmin_speed, vmax=vmax_speed, cmap='autumn', extent=extent)
        plt.title('Sound speed map (m/s)')
        plt.xlabel('X (mm)')
        plt.ylabel('Z (mm)')
        plt.colorbar()
        
        plt.subplot(122)
        plt.imshow(self.kmedium.density.T, vmin=vmin_density, vmax=vmax_density, cmap='summer', extent=extent)
        plt.title('Density map (kg/m^3)')
        plt.xlabel('X (mm)')
        plt.ylabel('Z (mm)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()