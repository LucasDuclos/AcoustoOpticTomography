import concurrent
import warnings
from AOT_biomaps.Config import config

from ._mainRecon import Recon
from .ReconEnums import ReconType, OptimizerType, ProcessType, SMatrixType, PotentialType, NoiseType
from .AOT_Optimizers import MLEM, LS, MAPEM, DEPIERRO, PDHG
from .AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from .AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from .AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

import os
import subprocess
import numpy as np
from datetime import datetime
from tempfile import gettempdir
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, List


# ============================================================================
# ALGORITHM FORMULAS (for error messages and documentation)
# ============================================================================

ALGORITHM_FORMULAS = {
    OptimizerType.MLEM: {
        "formula": r"theta^(k+1) = theta^(k) * (A^T * (y / (A*theta^(k) + epsilon))) / (A^T * 1)",
        "description": "Maximum Likelihood Expectation Maximization (multiplicative form)",
        "reference": "Shepp and Vardi, IEEE TMI, 1982",
        "required_params": ["denominatorThreshold"],
        "constraints": {
            "denominatorThreshold": "> 0",
            "numIterations": "> 0",
            "numSubsets": "> 0",
        },
        "notes": "If numSubsets > 1, becomes OSEM (Ordered Subset EM)"
    },
    OptimizerType.LS: {
        "formula": r"theta^(k+1) = theta^(k) - alpha * A^T * (A*theta^(k) - y)",
        "description": "Least Squares (Projected Gradient Descent)",
        "reference": "Landweber, 1951",
        "required_params": ["alpha"],
        "constraints": {
            "alpha": "> 0",
            "numIterations": "> 0",
        },
        "notes": "Convergence can be slow; consider using accelerated methods"
    },
    OptimizerType.MAPEM: {
        "formula": r"theta^(k+1) = theta^(k) * (A^T * (y / (A*theta^(k) + epsilon))) / (A^T * 1 + lambda * diag(H_U))",
        "description": "Maximum A Posteriori Expectation Maximization",
        "reference": "Green, IEEE TMI, 1990",
        "required_params": ["alpha", "beta"],
        "constraints": {
            "alpha": ">= 0",
            "beta": ">= 0",
            "denominatorThreshold": "> 0",
            "numIterations": "> 0",
        },
        "notes": "H_U is the Hessian diagonal of the potential function"
    },
    OptimizerType.DEPIERRO95: {
        "formula": r"theta^(k+1) = theta^(k) * (A^T * (y / (A*theta^(k) + epsilon))) / (A^T * 1 + sigma * beta * I)",
        "description": "De Pierro's quadratic regularization for EM",
        "reference": "De Pierro, IEEE TMI, 1995",
        "required_params": ["beta", "sigma"],
        "constraints": {
            "beta": ">= 0",
            "sigma": ">= 0",
            "denominatorThreshold": "> 0",
            "numIterations": "> 0",
        },
        "notes": "Convergent for MRF penalties"
    },
    OptimizerType.PPGMLEM: {
        "formula": r"theta^(k+1) = theta^(k) + alpha * (A^T * (y / (A*theta^(k) + epsilon) - 1)) / (A^T * 1 + beta * diag(H_U))",
        "description": "Penalized Preconditioned Gradient ML-EM",
        "reference": "Nuyts et al., IEEE TNS, 2002",
        "required_params": ["alpha", "beta"],
        "constraints": {
            "alpha": "> 0",
            "beta": ">= 0",
            "denominatorThreshold": "> 0",
            "numIterations": "> 0",
        },
        "notes": "Addresses numerical problems with large penalty strengths"
    },
    OptimizerType.PGC: {
        "formula": r"theta^(k+1) = theta^(k) + alpha * (A^T * (y / (A*theta^(k) + epsilon) - 1)) / (A^T * 1 + beta * diag(H_U))",
        "description": "Penalized Gauss-Newton Conjugate Gradient",
        "reference": "Nuyts et al., IEEE TNS, 2002",
        "required_params": ["alpha", "beta"],
        "constraints": {
            "alpha": "> 0",
            "beta": ">= 0",
            "denominatorThreshold": "> 0",
            "numIterations": "> 0",
        },
        "notes": "Requires penalty terms with derivative order >= 2"
    },
    OptimizerType.PDHG: {
        "formula": r"x^(k+1) = prox_{tau*TV}(x^(k) - tau * nabla_f(x^(k))) \ y^(k+1) = y^(k) + sigma * (A*x^(k+1) - y)",
        "description": "Primal-Dual Hybrid Gradient for TV regularization",
        "reference": "Chambolle and Pock, J. Math. Imaging Vis., 2011",
        "required_params": ["alpha"],
        "constraints": {
            "alpha": "> 0",
            "beta": ">= 0",
            "theta": "in [1.0, 2.0]",
            "k_security": "in (0, 1]",
            "numIterations": "> 0",
        },
        "notes": "tau and sigma are step sizes; prox is the proximal operator for TV"
    },
}


class AlgebraicRecon(Recon):
    """
    Algebraic reconstruction class for AOT_biomaps.
    
    This class provides a unified interface for all iterative reconstruction algorithms,
    including MLEM, LS, MAPEM, DEPIERRO95, PPGMLEM, PGC, and PDHG.
    
    Features:
    - Support for multiple optimizer types
    - Support for multiple system matrix types (DENSE, CSR, SELL)
    - Hyperparameter validation with detailed error messages including formulas
    - Backward compatibility with existing code
    
    Usage:
        # Basic MLEM reconstruction
        recon = AlgebraicRecon(
            optimizer=OptimizerType.MLEM,
            numIterations=1000,
            denominatorThreshold=1e-6
        )
        recon.generate_SMatrix()
        recon.run(processType=ProcessType.PYTHON, withTumor=True)
        
        # Bayesian reconstruction with regularization
        recon = AlgebraicRecon(
            optimizer=OptimizerType.MAPEM,
            numIterations=500,
            alpha=1.0, beta=0.1,
            potentialFunction=PotentialType.HUBER_PIECEWISE
        )
        recon.generate_SMatrix()
        recon.run(processType=ProcessType.PYTHON, withTumor=True)
        
        # Convex reconstruction with TV regularization
        recon = AlgebraicRecon(
            optimizer=OptimizerType.PDHG,
            numIterations=1000,
            alpha=0.1, beta=0.01, theta=1.0,
            k_security=0.8, use_power_method=True
        )
        recon.generate_SMatrix()
        recon.run(processType=ProcessType.PYTHON, withTumor=True)
    """
    def __init__(
        self,
        optimizer: OptimizerType = OptimizerType.MLEM,
        potentialFunction: Optional[PotentialType] = None,
        # Common parameters
        numIterations: int = 10000,
        numSubsets: int = 1,
        isSavingEachIteration: bool = True,
        maxSaves: int = 5000,
        denominatorThreshold: float = 1e-6,
        smatrixType: SMatrixType = SMatrixType.SELL,
        sparseThreshold: float = 0.1,
        isComplexeRecon: bool = False,
        device: Optional[str] = None,
        # Regularization parameters
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        sigma: Optional[float] = None,
        # PDHG-specific parameters
        theta: Optional[float] = None,
        L: Optional[float] = None,
        k_security: float = 0.8,
        use_power_method: bool = True,
        auto_alpha_gamma: float = 0.05,
        apply_positivity_clamp: bool = True,
        tikhonov_as_gradient: bool = False,
        use_laplacian: bool = True,
        laplacian_beta_scale: float = 1.0,
        # Potential function parameters
        corner: Optional[float] = None,
        face: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize AlgebraicRecon with specified parameters.
        
        Args:
            optimizer: Type of optimizer to use (OptimizerType enum)
            potentialFunction: Type of potential function for regularization (PotentialType enum)
            numIterations: Maximum number of iterations (default: 10000)
            numSubsets: Number of subsets for ordered subset algorithms (default: 1)
            isSavingEachIteration: Whether to save intermediate results (default: True)
            maxSaves: Maximum number of intermediate saves (default: 5000)
            denominatorThreshold: Threshold for denominator to avoid division by zero (default: 1e-6)
            smatrixType: Type of system matrix (default: SMatrixType.SELL)
            sparseThreshold: Threshold for sparse matrix construction (default: 0.1)
            isComplexeRecon: Whether to perform complex reconstruction (default: False)
            device: Device to use ('cpu' or 'gpu') (default: auto-detected)
            alpha: Regularization weight for MAPEM, DEPIERRO, PPGMLEM, PGC (default: None)
            beta: Regularization parameter for MAPEM, DEPIERRO, PPGMLEM, PGC, PDHG (default: None)
            gamma: Preconditioning parameter for PPGMLEM (default: None)
            delta: Huber threshold for MAPEM, PPGMLEM (default: None)
            sigma: Regularization scaling for DEPIERRO, PPGMLEM (default: None)
            theta: Relaxation parameter for PDHG (default: None)
            L: Spectral norm for PDHG (default: None)
            k_security: Security factor for PDHG step size (default: 0.8)
            use_power_method: Whether to use power method for spectral norm (default: True)
            auto_alpha_gamma: Gamma for auto alpha scaling (default: 0.05)
            apply_positivity_clamp: Whether to clamp to positive values (default: True)
            tikhonov_as_gradient: Whether to apply Tikhonov as gradient (default: False)
            use_laplacian: Whether to use Laplacian penalty (default: True)
            laplacian_beta_scale: Scaling factor for Laplacian beta (default: 1.0)
            corner: Corner parameter for potential functions (default: computed value)
            face: Face parameter for potential functions (default: computed value)
            **kwargs: Additional keyword arguments
        
        Raises:
            ValueError: If any hyperparameter fails validation
        """
        super().__init__(**kwargs)
        
        # Set reconstruction type
        self.reconType = ReconType.Algebraic
        self.optimizer = optimizer
        
        # Set potential function (default based on optimizer)
        if potentialFunction is not None:
            self.potentialFunction = potentialFunction
        else:
            # Default potential function based on optimizer
            if optimizer in (OptimizerType.PPGMLEM, OptimizerType.PGC, OptimizerType.DEPIERRO95):
                self.potentialFunction = PotentialType.HUBER_PIECEWISE
            else:
                self.potentialFunction = None
        
        # Store common parameters
        self.numIterations = numIterations
        self.numSubsets = numSubsets
        self.isSavingEachIteration = isSavingEachIteration
        self.maxSaves = maxSaves
        self.denominatorThreshold = denominatorThreshold
        self.isComplexeRecon = isComplexeRecon
        self.device = device if device is not None else config.select_best_gpu()
        self.SMatrix = None
        self.smatrixType = smatrixType
        self.sparseThreshold = sparseThreshold
        
        # Store regularization parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma
        
        # Store PDHG-specific parameters
        self.theta = theta
        self.L = L
        self.k_security = k_security
        self.use_power_method = use_power_method
        self.auto_alpha_gamma = auto_alpha_gamma
        self.apply_positivity_clamp = apply_positivity_clamp
        self.tikhonov_as_gradient = tikhonov_as_gradient
        self.use_laplacian = use_laplacian
        self.laplacian_beta_scale = laplacian_beta_scale
        
        # Set corner and face with defaults
        self.corner = corner if corner is not None else (0.5 - np.sqrt(2)/4) / np.sqrt(2)
        self.face = face if face is not None else 0.5 - np.sqrt(2)/4
        
        # Initialize reconstruction results
        self.reconPhantom: List[np.ndarray] = []
        self.reconLaser: List[np.ndarray] = []
        self.indices: List[int] = []
        self.MSE: Optional[List[float]] = None
        self.SSIM: Optional[List[float]] = None
        self.CRC: Optional[List[float]] = None
        
        # Validate hyperparameters
        self._validate_hyperparameters()
        
        # Handle complex reconstruction
        if self.isComplexeRecon:
            if self.experiment.AOsignal_withTumor is not None:
                self.experiment.AOsignal_withTumor_demodulated = self.experiment.parse_and_demodulate(withTumor=True)
            elif self.experiment.AOsignal_withoutTumor is not None:
                self.experiment.AOsignal_withoutTumor_demodulated = self.experiment.parse_and_demodulate(withTumor=False)
            else:
                raise ValueError("No AO signal available for demodulation. Please provide at least one signal, with or without tumor.")
            self.experiment.AcousticFields_demodulated = self.experiment.demodulate_acoustic_fields()
    
    def _validate_hyperparameters(self):
        """
        Validate all hyperparameters for the selected optimizer.
        
        If validation fails, raises ValueError with detailed message including:
        - The algorithm formula
        - The required hyperparameters
        - The constraints on each hyperparameter
        - The actual values provided
        
        Raises:
            ValueError: If any hyperparameter fails validation
        """
        # Get the formula information for the current optimizer
        if self.optimizer not in ALGORITHM_FORMULAS:
            warnings.warn(f"Unknown optimizer type: {self.optimizer}. Skipping hyperparameter validation.")
            return
        
        formula_info = ALGORITHM_FORMULAS[self.optimizer]
        errors = []
        
        # Check common parameters
        if self.numIterations <= 0:
            errors.append(f"numIterations must be > 0, got {self.numIterations}")
        if self.numSubsets <= 0:
            errors.append(f"numSubsets must be > 0, got {self.numSubsets}")
        if not isinstance(self.numIterations, int):
            errors.append(f"numIterations must be an integer, got {type(self.numIterations)}")
        if not isinstance(self.numSubsets, int):
            errors.append(f"numSubsets must be an integer, got {type(self.numSubsets)}")
        
        # Check optimizer-specific constraints
        constraints = formula_info.get("constraints", {})
        for param_name, constraint in constraints.items():
            param_value = getattr(self, param_name, None)
            if param_value is None:
                if param_name in formula_info.get("required_params", []):
                    errors.append(f"Required hyperparameter '{param_name}' is not set")
                continue
            
            if constraint == "> 0":
                if param_value <= 0:
                    errors.append(f"{param_name} must be > 0, got {param_value}")
            elif constraint == ">= 0":
                if param_value < 0:
                    errors.append(f"{param_name} must be >= 0, got {param_value}")
            elif constraint.startswith("in ["):
                interval = constraint[4:-1]
                if "," in interval:
                    low, high = map(float, interval.split(","))
                    if not (low <= param_value <= high):
                        errors.append(f"{param_name} must be in [{low}, {high}], got {param_value}")
            elif constraint.startswith("in ("):
                interval = constraint[4:-1]
                if "," in interval:
                    low, high = map(float, interval.split(","))
                    if not (low < param_value < high):
                        errors.append(f"{param_name} must be in ({low}, {high}), got {param_value}")
        
        # Special validation for PDHG
        if self.optimizer == OptimizerType.PDHG:
            if self.theta is not None and not (1.0 <= self.theta <= 2.0):
                errors.append(f"theta must be in [1.0, 2.0], got {self.theta}")
            if not (0 < self.k_security <= 1):
                errors.append(f"k_security must be in (0, 1], got {self.k_security}")
        
        # If there are errors, raise ValueError with detailed message
        if errors:
            error_msg = f"\n{'='*80}\n"
            error_msg += f"HYPERPARAMETER VALIDATION ERROR for {self.optimizer.value}\n"
            error_msg += f"{'='*80}\n\n"
            error_msg += f"Algorithm: {formula_info['description']}\n"
            error_msg += f"Reference: {formula_info.get('reference', 'N/A')}\n"
            error_msg += f"Formula: {formula_info['formula']}\n\n"
            error_msg += "Required hyperparameters:\n"
            for param_name in formula_info.get("required_params", []):
                param_value = getattr(self, param_name, None)
                param_desc = formula_info.get("constraints", {}).get(param_name, "")
                error_msg += f"  - {param_name}: {param_desc} (current: {param_value})\n"
            error_msg += f"\nConstraints:\n"
            for param_name, constraint in constraints.items():
                param_value = getattr(self, param_name, None)
                error_msg += f"  - {param_name}: {constraint} (current: {param_value})\n"
            if formula_info.get("notes"):
                error_msg += f"\nNotes:\n  {formula_info['notes']}\n"
            error_msg += f"\nErrors:\n"
            for error in errors:
                error_msg += f"  - {error}\n"
            error_msg += f"\n{'='*80}\n"
            raise ValueError(error_msg)

        

    # PUBLIC METHODS
    def generate_SMatrix(self, isShowLogs=True):
        print("Generating system matrix (processing acoustic fields)...")
        self.SMatrix = self._fillSMatrix(isShowLogs=True)

    def run(self, processType: ProcessType = ProcessType.PYTHON, withTumor: bool = True, show_logs: bool = True):
        """
        Run the algebraic reconstruction process.
        
        Dispatches to the appropriate reconstruction method based on processType.
        
        Args:
            processType: Type of processing (PYTHON or CASToR)
            withTumor: If True, reconstruct with tumor data; otherwise without
            show_logs: If True, display progress logs
            
        Raises:
            ValueError: If SMatrix is not generated or processType is unknown
        """
        if self.SMatrix is None:
            raise ValueError("System matrix (SMatrix) is not generated. Please call generate_SMatrix() before run().")
        
        if processType == ProcessType.CASToR:
            self._algebraic_recon_CASToR(withTumor=withTumor, show_logs=show_logs)
        elif processType == ProcessType.PYTHON:
            self._algebraic_recon_Python(withTumor=withTumor, show_logs=show_logs)
        else:
            raise ValueError(f"Unknown Algebraic reconstruction type: {processType}")

    def _algebraic_recon_Python(self, withTumor: bool = True, show_logs: bool = True):
        """
        Run algebraic reconstruction using Python implementation.
        
        Dispatches to the appropriate optimizer-specific method based on self.optimizer.
        
        Args:
            withTumor: If True, reconstruct with tumor data; otherwise without
            show_logs: If True, display progress logs
            
        Raises:
            ValueError: If the optimizer is not supported
        """
        # Check signal availability
        if withTumor:
            if self.experiment.AOsignal_withTumor is None:
                raise ValueError("AO signal with tumor is not available. Please generate AO signal with tumor in the experiment first.")
            y = self.experiment.AOsignal_withTumor
        else:
            if self.experiment.AOsignal_withoutTumor is None:
                raise ValueError("AO signal without tumor is not available. Please generate AO signal without tumor in the experiment first.")
            y = self.experiment.AOsignal_withoutTumor
        
        # Dispatch to optimizer-specific method
        if self.optimizer == OptimizerType.MLEM:
            self._run_MLEM(y=y, withTumor=withTumor, show_logs=show_logs)
        elif self.optimizer == OptimizerType.LS:
            self._run_LS(y=y, withTumor=withTumor, show_logs=show_logs)
        elif self.optimizer == OptimizerType.MAPEM:
            self._run_MAPEM(y=y, withTumor=withTumor, show_logs=show_logs)
        elif self.optimizer == OptimizerType.DEPIERRO95:
            self._run_DEPIERRO(y=y, withTumor=withTumor, show_logs=show_logs)
        elif self.optimizer == OptimizerType.PPGMLEM:
            self._run_PPGMLEM(y=y, withTumor=withTumor, show_logs=show_logs)
        elif self.optimizer == OptimizerType.PGC:
            self._run_PGC(y=y, withTumor=withTumor, show_logs=show_logs)
        elif self.optimizer == OptimizerType.PDHG:
            self._run_PDHG(y=y, withTumor=withTumor, show_logs=show_logs)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer}")

    def _algebraic_recon_CASToR(self, withTumor: bool = True, show_logs: bool = True):
        """
        Run algebraic reconstruction using CASToR.
        
        Args:
            withTumor: If True, reconstruct with tumor data; otherwise without
            show_logs: If True, display progress logs
            
        Raises:
            NotImplementedError: CASToR reconstruction is not yet implemented for all optimizers
        """
        # Define paths
        smatrix = os.path.join(self.saveDir, "system_matrix")
        if withTumor:
            fileName = 'AOSignals_withTumor.cdh'
        else:
            fileName = 'AOSignals_withoutTumor.cdh'

        # Check and generate input files if necessary
        if not os.path.isfile(os.path.join(self.saveDir, fileName)):
            if show_logs:
                print(f"Missing .cdh file. Generating {fileName}...")
            self.experiment.saveAOsignals_Castor(self.saveDir)

        # Check/generate system matrix
        if not os.path.isdir(smatrix):
            os.makedirs(smatrix, exist_ok=True)
        if not os.listdir(smatrix):
            if show_logs:
                print("System matrix missing. Generating...")
            self.experiment.saveAcousticFields(self.saveDir)

        # Verify that the .cdh file exists
        if not os.path.isfile(os.path.join(self.saveDir, fileName)):
            raise FileNotFoundError(f".cdh file does not exist: {fileName}")

        # Create output directory
        os.makedirs(os.path.join(self.saveDir, 'results', 'recon'), exist_ok=True)

        # Configure environment for CASToR
        env = os.environ.copy()
        env.update({
            "CASTOR_DIR": self.experiment.params.reconstruction['castor_executable'],
            "CASTOR_CONFIG": os.path.join(self.experiment.params.reconstruction['castor_executable'], "config"),
            "CASTOR_64bits": "1",
            "CASTOR_OMP": "1",
            "CASTOR_SIMD": "1",
            "CASTOR_ROOT": "1",
        })

        # Build command
        cmd = [
            os.path.join(self.experiment.params.reconstruction['castor_executable'], "bin", "castor-recon"),
            "-df", os.path.join(self.saveDir, fileName),
            "-opti", self.optimizer.value,
            "-it", f"{self.numIterations}:{self.numSubsets}",
            "-proj", "matrix",
            "-dout", os.path.join(self.saveDir, 'results', 'recon'),
            "-th", str(os.cpu_count()),
            "-vb", "5",
            "-proj-comp", "1",
            "-ignore-scanner",
            "-data-type", "AOT",
            "-ignore-corr", "cali,fdur",
            "-system-matrix", smatrix,
        ]

        # Add optimizer-specific parameters
        if self.optimizer == OptimizerType.MLEM:
            pass  # No additional parameters needed
        elif self.optimizer == OptimizerType.LS:
            if self.alpha is not None:
                cmd.extend(["-alpha", str(self.alpha)])
        elif self.optimizer == OptimizerType.MAPEM:
            if self.alpha is not None:
                cmd.extend(["-alpha", str(self.alpha)])
            if self.beta is not None:
                cmd.extend(["-beta", str(self.beta)])
        elif self.optimizer == OptimizerType.DEPIERRO95:
            if self.beta is not None:
                cmd.extend(["-beta", str(self.beta)])
            if self.sigma is not None:
                cmd.extend(["-sigma", str(self.sigma)])
        elif self.optimizer == OptimizerType.PPGMLEM:
            if self.alpha is not None:
                cmd.extend(["-alpha", str(self.alpha)])
            if self.beta is not None:
                cmd.extend(["-beta", str(self.beta)])
            if self.gamma is not None:
                cmd.extend(["-gamma", str(self.gamma)])
            if self.delta is not None:
                cmd.extend(["-delta", str(self.delta)])
        elif self.optimizer == OptimizerType.PGC:
            if self.alpha is not None:
                cmd.extend(["-alpha", str(self.alpha)])
            if self.beta is not None:
                cmd.extend(["-beta", str(self.beta)])
        elif self.optimizer == OptimizerType.PDHG:
            if self.alpha is not None:
                cmd.extend(["-alpha", str(self.alpha)])
            if self.beta is not None:
                cmd.extend(["-beta", str(self.beta)])
            if self.theta is not None:
                cmd.extend(["-theta", str(self.theta)])
            if self.L is not None:
                cmd.extend(["-L", str(self.L)])

        # Display command for debugging
        if show_logs:
            print("CASToR command:")
            print(" ".join(cmd))

        # Path to temporary script
        recon_script_path = os.path.join(gettempdir(), 'recon.sh')

        # Write bash script
        with open(recon_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"export PATH={env['CASTOR_DIR']}/bin:$PATH\n")
            f.write(f"export LD_LIBRARY_PATH={env['CASTOR_DIR']}/lib:$LD_LIBRARY_PATH\n")
            f.write(" ".join(cmd) + "\n")

        # Make script executable and run it
        subprocess.run(["chmod", "+x", recon_script_path], check=True)
        if show_logs:
            print(f"Running reconstruction with CASToR...")
        result = subprocess.run(recon_script_path, env=env, check=True, capture_output=True, text=True)

        # Display CASToR output for debugging
        if show_logs:
            print("CASToR output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)

        if show_logs:
            print("Reconstruction completed successfully.")
        self.load_reconCASToR(withTumor=withTumor)

    def _run_MLEM(self, y, withTumor: bool = True, show_logs: bool = True):
        """Run MLEM reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, _ = MLEM(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                device=self.device,
                denominator_threshold=self.denominatorThreshold,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                smatrixType=self.smatrixType,
            )
        else:
            self.reconLaser, self.indices, _ = MLEM(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                device=self.device,
                denominator_threshold=self.denominatorThreshold,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                smatrixType=self.smatrixType,
            )

    def _run_LS(self, y, withTumor: bool = True, show_logs: bool = True):
        """Run Least Squares reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, _ = LS(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                device=self.device,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                smatrixType=self.smatrixType,
                alpha=self.alpha,
            )
        else:
            self.reconLaser, self.indices, _ = LS(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                device=self.device,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                smatrixType=self.smatrixType,
                alpha=self.alpha,
            )

    def _run_MAPEM(self, y, withTumor: bool = True, show_logs: bool = True):
        """Run MAPEM reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices = MAPEM(
                SMatrix=self.SMatrix,
                y=y,
                potential_type=self.potentialFunction,
                alpha=self.alpha if self.alpha is not None else 1.0,
                beta=self.beta if self.beta is not None else 1.0,
                delta=self.delta if self.delta is not None else 0.01,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                device=self.device,
                smatrixType=self.smatrixType,
            )
        else:
            self.reconLaser, self.indices = MAPEM(
                SMatrix=self.SMatrix,
                y=y,
                potential_type=self.potentialFunction,
                alpha=self.alpha if self.alpha is not None else 1.0,
                beta=self.beta if self.beta is not None else 1.0,
                delta=self.delta if self.delta is not None else 0.01,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                device=self.device,
                smatrixType=self.smatrixType,
            )

    def _run_DEPIERRO(self, y, withTumor: bool = True, show_logs: bool = True):
        """Run DEPIERRO95 reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices = DEPIERRO(
                SMatrix=self.SMatrix,
                y=y,
                beta=self.beta if self.beta is not None else 1.0,
                sigma=self.sigma if self.sigma is not None else 1.0,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                device=self.device,
                smatrixType=self.smatrixType,
            )
        else:
            self.reconLaser, self.indices = DEPIERRO(
                SMatrix=self.SMatrix,
                y=y,
                beta=self.beta if self.beta is not None else 1.0,
                sigma=self.sigma if self.sigma is not None else 1.0,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                device=self.device,
                smatrixType=self.smatrixType,
            )

    def _run_PPGMLEM(self, y, withTumor: bool = True, show_logs: bool = True):
        """Run PPGMLEM reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices = MAPEM(
                SMatrix=self.SMatrix,
                y=y,
                potential_type=self.potentialFunction,
                alpha=self.alpha if self.alpha is not None else 1.0,
                beta=self.beta if self.beta is not None else 1.0,
                delta=self.delta if self.delta is not None else 0.01,
                gamma=self.gamma if self.gamma is not None else 0.01,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                device=self.device,
                smatrixType=self.smatrixType,
            )
        else:
            self.reconLaser, self.indices = MAPEM(
                SMatrix=self.SMatrix,
                y=y,
                potential_type=self.potentialFunction,
                alpha=self.alpha if self.alpha is not None else 1.0,
                beta=self.beta if self.beta is not None else 1.0,
                delta=self.delta if self.delta is not None else 0.01,
                gamma=self.gamma if self.gamma is not None else 0.01,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                device=self.device,
                smatrixType=self.smatrixType,
            )

    def _run_PGC(self, y, withTumor: bool = True, show_logs: bool = True):
        """Run PGC reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices = MAPEM(
                SMatrix=self.SMatrix,
                y=y,
                potential_type=self.potentialFunction,
                alpha=self.alpha if self.alpha is not None else 1.0,
                beta=self.beta if self.beta is not None else 1.0,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                device=self.device,
                smatrixType=self.smatrixType,
            )
        else:
            self.reconLaser, self.indices = MAPEM(
                SMatrix=self.SMatrix,
                y=y,
                potential_type=self.potentialFunction,
                alpha=self.alpha if self.alpha is not None else 1.0,
                beta=self.beta if self.beta is not None else 1.0,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                device=self.device,
                smatrixType=self.smatrixType,
            )

    def _run_PDHG(self, y, withTumor: bool = True, show_logs: bool = True):
        """Run PDHG reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices = PDHG(
                SMatrix=self.SMatrix,
                y=y,
                alpha=self.alpha,
                beta=self.beta,
                theta=self.theta,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                L=self.L,
                withTumor=withTumor,
                device=self.device,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                smatrixType=self.smatrixType,
                k_security=self.k_security,
                use_power_method=self.use_power_method,
                auto_alpha_gamma=self.auto_alpha_gamma,
                apply_positivity_clamp=self.apply_positivity_clamp,
                tikhonov_as_gradient=self.tikhonov_as_gradient,
                use_laplacian=self.use_laplacian,
                laplacian_beta_scale=self.laplacian_beta_scale,
            )
        else:
            self.reconLaser, self.indices = PDHG(
                SMatrix=self.SMatrix,
                y=y,
                alpha=self.alpha,
                beta=self.beta,
                theta=self.theta,
                numIterations=self.numIterations,
                isSavingEachIteration=self.isSavingEachIteration,
                L=self.L,
                withTumor=withTumor,
                device=self.device,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                smatrixType=self.smatrixType,
                k_security=self.k_security,
                use_power_method=self.use_power_method,
                auto_alpha_gamma=self.auto_alpha_gamma,
                apply_positivity_clamp=self.apply_positivity_clamp,
                tikhonov_as_gradient=self.tikhonov_as_gradient,
                use_laplacian=self.use_laplacian,
                laplacian_beta_scale=self.laplacian_beta_scale,
            )
    
    def plot_MSE(self, isSaving=True, log_scale_x=False, log_scale_y=False, figSize=(4,3), show_logs=True):
        """
        Plot the Mean Squared Error (MSE) of the reconstruction.

        Parameters:
            isSaving: bool, whether to save the plot.
            log_scale_x: bool, if True, use logarithmic scale for the x-axis.
            log_scale_y: bool, if True, use logarithmic scale for the y-axis.
        Returns:
            None
        """
        if not self.MSE:
            raise ValueError("MSE is empty. Please calculate MSE first.")

        best_idx = self.indices[np.argmin(self.MSE)]
        if show_logs:
            print(f"Lowest MSE = {np.min(self.MSE):.4f} at iteration {best_idx+1}")
        # Plot MSE curve
        plt.figure(figsize=figSize)
        plt.plot(self.indices, self.MSE, 'r-', label="MSE curve")
        # Add blue dashed lines
        plt.axhline(np.min(self.MSE), color='blue', linestyle='--', label=f"Min MSE = {np.min(self.MSE):.4f}")
        plt.axvline(best_idx, color='blue', linestyle='--', label=f"Iteration = {best_idx+1}")
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title("MSE vs. Iteration")
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.tight_layout()
        if isSaving and self.saveDir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            scale_str = ""
            if log_scale_x and log_scale_y:
                scale_str = "_loglog"
            elif log_scale_x:
                scale_str = "_logx"
            elif log_scale_y:
                scale_str = "_logy"
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_MSE_plot_{self.optimizer.name}_{scale_str}{date_str}.png')
            plt.savefig(SavingFolder, dpi=300)
            if show_logs:
                print(f"MSE plot saved to {SavingFolder}")

        plt.show()

    def show_MSE_bestRecon(self, isSaving=True, show_logs=True, figSize=(15, 5)):
        if not self.MSE:
            raise ValueError("MSE is empty. Please calculate MSE first.")

        best_idx = np.argmin(self.MSE)
        best_recon = self.reconPhantom[best_idx]

        # Crée la figure et les axes
        fig, axs = plt.subplots(1, 3, figsize=figSize)

        # Left: Best reconstructed image (normalized)
        im0 = axs[0].imshow(best_recon,
                            extent=(self.experiment.params.general['Xrange'][0]*1000, self.experiment.params.general['Xrange'][1]*1000,
                                    self.experiment.params.general['Zrange'][1]*1000, self.experiment.params.general['Zrange'][0]*1000),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[0].set_title(f"Min MSE Reconstruction\nIter {self.indices[best_idx]}, MSE={np.min(self.MSE):.4f}")
        axs[0].set_xlabel("x (mm)", fontsize=12)
        axs[0].set_ylabel("z (mm)", fontsize=12)
        axs[0].tick_params(axis='both', which='major', labelsize=8)

        # Middle: Ground truth (normalized)
        im1 = axs[1].imshow(self.experiment.OpticImage.phantom,
                            extent=(self.experiment.params.general['Xrange'][0]*1000, self.experiment.params.general['Xrange'][1]*1000,
                                    self.experiment.params.general['Zrange'][1]*1000, self.experiment.params.general['Zrange'][0]*1000),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[1].set_title(r"Ground Truth ($\lambda$)")
        axs[1].set_xlabel("x (mm)", fontsize=12)
        axs[1].set_ylabel("z (mm)", fontsize=12)
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # Right: Reconstruction at last iteration
        lastRecon = self.reconPhantom[-1]
        if self.experiment.OpticImage.phantom.shape != lastRecon.shape:
            lastRecon = lastRecon.T
        im2 = axs[2].imshow(lastRecon,
                            extent=(self.experiment.params.general['Xrange'][0]*1000, self.experiment.params.general['Xrange'][1]*1000,
                                    self.experiment.params.general['Zrange'][1]*1000, self.experiment.params.general['Zrange'][0]*1000),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[2].set_title(f"Last Reconstruction\nIter {self.numIterations * self.numSubsets}, MSE={np.mean((self.experiment.OpticImage.phantom - lastRecon) ** 2):.4f}")
        axs[2].set_xlabel("x (mm)", fontsize=12)
        axs[2].set_ylabel("z (mm)", fontsize=12)
        axs[2].tick_params(axis='both', which='major', labelsize=8)

        # Ajoute une colorbar horizontale centrée en dessous des trois plots
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Normalized Intensity', fontsize=12)
        cbar.ax.tick_params(labelsize=8)

        plt.subplots_adjust(wspace=0.3)

        if isSaving and self.saveDir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            savePath = os.path.join(self.saveDir, 'results')
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_comparison_MSE_BestANDLastRecon_{self.optimizer.name}_{date_str}.png')
            plt.savefig(SavingFolder, dpi=300, bbox_inches='tight')
            if show_logs:
                print(f"MSE plot saved to {SavingFolder}")

        plt.show()

    def show_theta_animation(self, vmin=None, vmax=None, total_duration_ms=3000, save_path=None, max_frames=1000, figSize=(4, 4), isPropMSE=True, show_logs=True):
        """
        Show theta iteration animation with speed proportional to MSE acceleration.
        In "propMSE" mode: slow down when MSE changes rapidly, speed up when MSE stagnates.

        Parameters:
            vmin, vmax: color limits (optional)
            total_duration_ms: total duration of the animation in milliseconds
            save_path: path to save animation (e.g., 'theta.gif')
            max_frames: maximum number of frames to include (default: 1000)
            isPropMSE: if True, use adaptive speed based on MSE (default: True)
        """
        import matplotlib as mpl
        mpl.rcParams['animation.embed_limit'] = 200

        if len(self.reconPhantom) == 0 or len(self.reconPhantom) < 2:
            raise ValueError("Not enough theta matrices available for animation.")

        if isPropMSE and (self.MSE is None or len(self.MSE) == 0):
            raise ValueError("MSE is empty or not calculated. Please calculate MSE first.")

        frames = np.array(self.reconPhantom)
        mse = np.array(self.MSE)

        # Sous-échantillonnage initial
        step = max(1, len(frames) // max_frames)
        frames_subset = frames[::step]
        indices_subset = self.indices[::step]
        mse_subset = mse[::step]

        if vmin is None:
            vmin = np.min(frames_subset)
        if vmax is None:
            vmax = np.max(frames_subset)

        fig, ax = plt.subplots(figsize=figSize, dpi=100)
        im = ax.imshow(
            frames_subset[0],
            extent=(
                self.experiment.params.general['Xrange'][0],
                self.experiment.params.general['Xrange'][1],
                self.experiment.params.general['Zrange'][1],
                self.experiment.params.general['Zrange'][0]
            ),
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            cmap='hot'
        )
        title = ax.set_title(f"Iteration {indices_subset[0]}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        plt.tight_layout()

        if isPropMSE:
            # Calcule la dérivée première (variation du MSE)
            mse_diff = np.gradient(mse_subset)
            # Calcule la dérivée seconde (accélération du MSE)
            mse_accel = np.gradient(mse_diff)
            # Normalise l'accélération entre 0 et 1 (en valeur absolue)
            mse_accel_normalized = np.abs(mse_accel)
            mse_accel_normalized /= (np.max(mse_accel_normalized) + 1e-10)

            # Prépare les frames pour le mode "propMSE"
            all_frames = []
            all_indices = []

            for i in range(len(frames_subset)):
                # Nombre de duplications inversement proportionnel à l'accélération (pour ralentir quand MSE change vite)
                # Plus l'accélération est élevée, plus on duplique (pour ralentir)
                num_duplicates = max(1, int(1 + 9 * mse_accel_normalized[i]))
                all_frames.extend([frames_subset[i]] * num_duplicates)
                all_indices.extend([indices_subset[i]] * num_duplicates)

            # Ajuste le nombre total de frames pour respecter la durée
            target_frames = int(total_duration_ms / 10)  # 10 ms par frame
            if len(all_frames) > target_frames:
                step_prop = len(all_frames) // target_frames
                all_frames = all_frames[::step_prop]
                all_indices = all_indices[::step_prop]

        else:  # Mode "linéaire"
            all_frames = frames_subset
            all_indices = indices_subset

        def update(frame_idx):
            im.set_array(all_frames[frame_idx])
            title.set_text(f"Iteration {all_indices[frame_idx]}")
            return [im, title]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(all_frames),
            interval=10,  # 10 ms par frame
            blit=False,
        )

        if save_path:
            if save_path.endswith(".gif"):
                ani.save(save_path, writer=animation.PillowWriter(fps=100))
            elif save_path.endswith(".mp4"):
                ani.save(save_path, writer="ffmpeg", fps=30)
            if show_logs:
                print(f"Animation saved to {save_path}")

        plt.close(fig)
        return HTML(ani.to_jshtml())

    def plot_SSIM(self, isSaving=True, log_scale_x=False, log_scale_y=False, figSize=(4,3), show_logs=True):
        if not self.SSIM:
            raise ValueError("SSIM is empty. Please calculate SSIM first.")

        best_idx = self.indices[np.argmax(self.SSIM)]
        if show_logs:
            print(f"Highest SSIM = {np.max(self.SSIM):.4f} at iteration {best_idx+1}")
        # Plot SSIM curve
        plt.figure(figsize=figSize)
        plt.plot(self.indices, self.SSIM, 'r-', label="SSIM curve")
        # Add blue dashed lines
        plt.axhline(np.max(self.SSIM), color='blue', linestyle='--', label=f"Max SSIM = {np.max(self.SSIM):.4f}")
        plt.axvline(best_idx, color='blue', linestyle='--', label=f"Iteration = {best_idx}")
        plt.xlabel("Iteration")
        plt.ylabel("SSIM")
        plt.title("SSIM vs. Iteration")
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.tight_layout()
        if isSaving and self.saveDir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            scale_str = ""
            if log_scale_x and log_scale_y:
                scale_str = "_loglog"
            elif log_scale_x:
                scale_str = "_logx"
            elif log_scale_y:
                scale_str = "_logy"
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_SSIM_plot_{self.optimizer.name}_{scale_str}{date_str}.png')
            plt.savefig(SavingFolder, dpi=300)
            if show_logs:
                print(f"SSIM plot saved to {SavingFolder}")

        plt.show()

    def show_SSIM_bestRecon(self, isSaving=True, figSize=(15, 5), show_logs=True):
        
        if not self.SSIM:
            raise ValueError("SSIM is empty. Please calculate SSIM first.")

        best_idx = np.argmax(self.SSIM)
        best_recon = self.reconPhantom[best_idx]

        # ----------------- Plotting -----------------
        _, axs = plt.subplots(1, 3, figsize=figSize)  # 1 row, 3 columns

        # Left: Best reconstructed image (normalized)
        im0 = axs[0].imshow(best_recon, 
                            extent=(self.experiment.params.general['Xrange'][0], self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[0].set_title(f"Max SSIM Reconstruction\nIter {self.indices[best_idx]}, SSIM={np.min(self.MSE):.4f}")
        axs[0].set_xlabel("x (mm)")
        axs[0].set_ylabel("z (mm)")
        plt.colorbar(im0, ax=axs[0])

        # Middle: Ground truth (normalized)
        im1 = axs[1].imshow(self.experiment.OpticImage.laser.intensity, 
                            extent=(self.experiment.params.general['Xrange'][0], self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[1].set_title(r"Ground Truth ($\lambda$)")
        axs[1].set_xlabel("x (mm)")
        axs[1].set_ylabel("z (mm)")
        plt.colorbar(im1, ax=axs[1])

        # Right: Reconstruction at iter 350
        lastRecon = self.reconPhantom[-1] 
        im2 = axs[2].imshow(lastRecon,
                            extent=(self.experiment.params.general['Xrange'][0], self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[2].set_title(f"Last Reconstruction\nIter {self.numIterations * self.numSubsets}, SSIM={self.SSIM[-1]:.4f}")
        axs[2].set_xlabel("x (mm)")
        axs[2].set_ylabel("z (mm)")
        plt.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        if isSaving:
            now = datetime.now()    
            date_str = now.strftime("%Y_%d_%m_%y")
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_comparison_SSIM_BestANDLastRecon_{self.optimizer.name}_{date_str}.png')
            plt.savefig(SavingFolder, dpi=300)
            if show_logs:
                print(f"SSIM plot saved to {SavingFolder}")
        plt.show()

    def plot_CRC_vs_Noise(self, use_ROI=True, fin=None, min_distance=0.01, figSize = (4,3),
                     log_scale_x=False, log_scale_y=False, isSaving=False, show_logs=True):
        """
        Plot CRC vs Noise with min_distance always calculated in linear space.

        Args:
            min_distance: Minimum linear distance between points (always calculated in linear space)
            log_scale_x: Display X axis in logarithmic scale (but distance calculation remains linear)
            log_scale_y: Display Y axis in logarithmic scale
        """
        # Vérifications initiales
        if self.reconLaser is None or self.reconLaser == []:
            raise ValueError("Reconstructed laser is empty. Run reconstruction first.")
        if isinstance(self.reconLaser, list) and len(self.reconLaser) == 1:
            raise ValueError("Reconstructed Image without tumor is a single frame. Run with isSavingEachIteration=True.")
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")
        if isinstance(self.reconPhantom, list) and len(self.reconPhantom) == 1:
            raise ValueError("Reconstructed Image with tumor is a single frame. Run with isSavingEachIteration=True.")

        if fin is None:
            fin = len(self.reconPhantom) - 1
        iter_range = self.indices[:fin+1]

        if self.CRC is None:
            self.calculateCRC(use_ROI=use_ROI)

        # Calcul des valeurs de bruit
        noise_values = []
        for i in range(len(iter_range)):
            recon_without_tumor = self.reconLaser[i]
            noise = np.mean(np.abs(recon_without_tumor - self.experiment.OpticImage.laser.intensity))
            noise_values.append(max(noise, 1e-10))  # Évite les valeurs nulles

        # Sous-échantillonnage TOUJOURS basé sur la distance linéaire
        sampled_indices = [0]
        for i in range(1, len(noise_values)):
            last_noise = noise_values[sampled_indices[-1]]
            current_noise = noise_values[i]

            # Calcul de la distance EN LINÉAIRE (peu importe l'échelle d'affichage)
            distance = abs(current_noise - last_noise)

            if distance > min_distance:
                sampled_indices.append(i)

        sampled_indices.append(len(noise_values) - 1)

        # Vérification si l'avant-dernier point est trop proche du dernier
        if len(sampled_indices) > 1:
            last_idx = sampled_indices[-1]
            prev_idx = sampled_indices[-2]
            last_noise = noise_values[last_idx]
            prev_noise = noise_values[prev_idx]

            if abs(last_noise - prev_noise) <= min_distance:
                sampled_indices = sampled_indices[:-1]  # Supprime l'avant-dernier

        # Extraction des données
        sampled_noise = [noise_values[i] for i in sampled_indices]
        sampled_CRC = [self.CRC[i] for i in sampled_indices]
        sampled_iter = [iter_range[i] for i in sampled_indices]

        # Création de la figure
        plt.figure(figsize=figSize)
        plt.plot(sampled_noise, sampled_CRC, 'o-', color='blue', label=f'{self.optimizer.name}')

        # Positionnement des labels
        for x, y, it in zip(sampled_noise, sampled_CRC, sampled_iter):
            plt.text(x * 1.01, y * 1.01, str(it),
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        # Configuration des axes
        plt.xlabel("Noise (mean absolute error)")
        plt.ylabel("CRC (Contrast Recovery Coefficient)")
        plt.title(f"CRC vs Noise (linear min_distance={min_distance})")

        # Application des échelles (uniquement pour l'affichage)
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')

        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()

        # Sauvegarde
        if isSaving:
            if self.saveDir is None:
                print("Warning: saveDir is None. Configure saving path to save the figure.")
            else:
                os.makedirs(self.saveDir, exist_ok=True)
                now = datetime.now()
                date_str = now.strftime("%Y_%m_%d_%H%M")
                scale_desc = "logX" if log_scale_x else "linX"
                scale_desc += "_logY" if log_scale_y else "_linY"
                filename = f"CRCvsNOISE_{len(self.experiment.AcousticFields)}scans_{self.optimizer.name}_{scale_desc}_{date_str}.png"
                save_path = os.path.join(self.saveDir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if show_logs:
                    print(f"Plot saved to: {save_path}")

        plt.tight_layout()
        plt.show()

    def show_reconstruction_progress(self, start=0, fin=None, save_path=None, with_tumor=True, show_logs=True):
        """
        Show the reconstruction progress for either with or without tumor.
        If isPropMSE is True, the frame selection is adapted to MSE changes.
        Otherwise, indices are evenly spaced between start and fin.

        Parameters:
            start: int, starting iteration index
            fin: int, ending iteration index (inclusive)
            duration: int, duration of the animation in milliseconds
            save_path: str, path to save the figure (optional)
            with_tumor: bool, if True, show reconstruction with tumor; else without (default: True)
            isPropMSE: bool, if True, use adaptive speed based on MSE (default: True)
        """
        import matplotlib as mpl
        mpl.rcParams['animation.embed_limit'] = 200

        if fin is None:
            fin = len(self.reconPhantom) - 1 if with_tumor else len(self.reconLaser) - 1

        # Check data availability
        if with_tumor:
            if self.reconPhantom is None or self.reconPhantom == []:
                raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")
            if isinstance(self.reconPhantom, list) and len(self.reconPhantom) == 1:
                raise ValueError("Reconstructed Image with tumor is a single frame. Run reconstruction with isSavingEachIteration=True.")
            recon_list = self.reconPhantom
            ground_truth = self.experiment.OpticImage.phantom
            title_suffix = "with_tumor"
        else:
            if self.reconLaser is None or self.reconLaser == []:
                raise ValueError("Reconstructed laser is empty. Run reconstruction first.")
            if isinstance(self.reconLaser, list) and len(self.reconLaser) == 1:
                raise ValueError("Reconstructed Image without tumor is a single frame. Run reconstruction with isSavingEachIteration=True.")
            recon_list = self.reconLaser
            ground_truth = self.experiment.OpticImage.laser.intensity
            title_suffix = "without_tumor"

        # Collect data for all iterations
        recon_list_data = []
        diff_abs_list = []
        mse_list = []
        noise_list = []

        for i in range(start, fin + 1):
            recon = recon_list[i]
            diff_abs = np.abs(recon - ground_truth)
            mse = np.mean((ground_truth.flatten() - recon.flatten())**2)
            noise = np.mean(np.abs(recon - ground_truth))

            recon_list_data.append(recon)
            diff_abs_list.append(diff_abs)
            mse_list.append(mse)
            noise_list.append(noise)

        # Calculate global min/max for difference images
        global_min_diff = np.min([d.min() for d in diff_abs_list[1:]])
        global_max_diff = np.max([d.max() for d in diff_abs_list[1:]])

        # Evenly spaced indices
        num_frames = min(5, fin - start + 1)
        all_indices = np.linspace(start, fin, num_frames, dtype=int).tolist()

        # Plot
        nrows = min(5, len(all_indices))
        ncols = 3  # Recon, |Recon - GT|, Ground Truth
        vmin, vmax = 0, 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))

        for i, iter_idx in enumerate(all_indices[:nrows]):
            idx_in_list = iter_idx - start  # Index in the collected data lists
            recon = recon_list_data[idx_in_list]
            diff_abs = diff_abs_list[idx_in_list]
            mse_val = mse_list[idx_in_list]
            noise = noise_list[idx_in_list]

            im0 = axs[i, 0].imshow(recon, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')
            axs[i, 0].set_title(f"Reconstruction\nIter {self.indices[iter_idx]}, MSE={mse_val:.2e}", fontsize=10)
            axs[i, 0].axis('off')
            plt.colorbar(im0, ax=axs[i, 0])

            im1 = axs[i, 1].imshow(diff_abs, cmap='viridis',
                                vmin=global_min_diff,
                                vmax=global_max_diff,
                                aspect='equal')
            axs[i, 1].set_title(f"|Recon - Ground Truth|\nNoise={noise:.2e}", fontsize=10)
            axs[i, 1].axis('off')
            plt.colorbar(im1, ax=axs[i, 1])

            im2 = axs[i, 2].imshow(ground_truth, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')
            axs[i, 2].set_title(r"Ground Truth", fontsize=10)
            axs[i, 2].axis('off')
            plt.colorbar(im2, ax=axs[i, 2])

        plt.tight_layout()

        if save_path:
            # Add suffix to filename based on with_tumor parameter
            if '.' in save_path:
                name, ext = save_path.rsplit('.', 1)
                save_path = f"{name}_{title_suffix}.{ext}"
            else:
                save_path = f"{save_path}_{title_suffix}"
            plt.savefig(save_path, dpi=300)
            if show_logs:
                print(f"Figure saved to: {save_path}")

        plt.show()

    def check_existing_file(self, date=None, withTumor=True):
        """
        Check if the reconstruction file already exists, based on current instance parameters.

        Args:
            date (str, optional): Date string in format "ddmm". If None, uses current date.
            withTumor (bool): If True, checks reconPhantom.npy; otherwise, checks reconLaser.npy.
            overwrite (bool): If True, ignores existing files and returns True for saving.

        Returns:
            tuple: (bool: whether to save, str: the filepath)
        """
        if self.saveDir is None:
            raise ValueError("Save directory is not specified.")
        if date is None:
            date = datetime.now().strftime("%d%m")
        results_dir = os.path.join(self.saveDir, f'results_{date}_{self.optimizer.value}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Détermine le nom du fichier en fonction de withTumor
        indices_file = os.path.join(results_dir, f"indices_{'withTumor' if withTumor else 'withoutTumor'}.npy")

        # Si le fichier existe retourne True
        if os.path.exists(indices_file):
            return (True, results_dir)

        # Sinon, retourne False, indiquant qu'on peut sauvegarder
        return (False, results_dir)

    def load(self, withTumor=True, results_date=None, optimizer=None, filePath=None, show_logs=True):
        """
        Load the reconstruction results (reconPhantom or reconLaser) and indices as lists of 2D np arrays into self.
        If the loaded file is a 3D array, it is split into a list of 2D arrays.
        Args:
            withTumor: If True, loads reconPhantom (with tumor), else reconLaser (without tumor).
            results_date: Date string (format "ddmm") to specify which results to load. If None, uses the most recent date in saveDir.
            optimizer: Optimizer name (as string or enum) to filter results. If None, uses the current optimizer of the instance.
            filePath: Optional. If provided, loads directly from this path (overrides saveDir and results_date).
        """
        if filePath is not None:
            # Mode chargement direct depuis un fichier
            recon_key = 'reconPhantom' if withTumor else 'reconLaser'
            recon_path = filePath
            if not os.path.exists(recon_path):
                raise FileNotFoundError(f"No reconstruction file found at {recon_path}.")
            # Charge le fichier (3D ou liste de 2D)
            data = np.load(recon_path, allow_pickle=True)
            # Découpe en liste de 2D si c'est un tableau 3D
            if isinstance(data, np.ndarray) and data.ndim == 3:
                if withTumor:
                    self.reconPhantom = [data[i, :, :] for i in range(data.shape[0])]
                else:
                    self.reconLaser = [data[i, :, :] for i in range(data.shape[0])]
            else:
                # Sinon, suppose que c'est déjà une liste de 2D
                if withTumor:
                    self.reconPhantom = data
                else:
                    self.reconLaser = data
            # Essayer de charger les indices
            base_dir, _ = os.path.split(recon_path)
            indices_path = os.path.join(base_dir, 'indices.npy')
            if os.path.exists(indices_path):
                indices_data = np.load(indices_path, allow_pickle=True)
                if isinstance(indices_data, np.ndarray) and indices_data.ndim == 3:
                    self.indices = [indices_data[i, :, :] for i in range(indices_data.shape[0])]
                else:
                    self.indices = indices_data
            else:
                self.indices = None
                
            if show_logs:
                print(f"Loaded reconstruction results and indices from {recon_path}")
        else:
            # Mode chargement depuis le répertoire de résultats
            if self.saveDir is None:
                raise ValueError("Save directory is not specified. Please set saveDir before loading.")
            # Use current optimizer and potential function if not provided
            opt_name = optimizer.value if optimizer is not None else self.optimizer.value
            # Build the base directory pattern
            dir_pattern = f'results_*_{opt_name}'
            # Add parameters to the pattern based on the optimizer
            if optimizer is None:
                optimizer = self.optimizer
            if optimizer == OptimizerType.PPGMLEM:
                beta_str = f'_Beta_{self.beta}'
                delta_str = f'_Delta_{self.delta}'
                gamma_str = f'_Gamma_{self.gamma}'
                sigma_str = f'_Sigma_{self.sigma}'
                dir_pattern += f'{beta_str}{delta_str}{gamma_str}{sigma_str}'
            elif optimizer in (OptimizerType.PGC, OptimizerType.DEPIERRO95):
                beta_str = f'_Beta_{self.beta}'
                sigma_str = f'_Sigma_{self.sigma}'
                dir_pattern += f'{beta_str}{sigma_str}'
            # Find the most recent results directory if no date is specified
            if results_date is None:
                dirs = [d for d in os.listdir(self.saveDir) if os.path.isdir(os.path.join(self.saveDir, d)) and dir_pattern in d]
                if not dirs:
                    raise FileNotFoundError(f"No matching results directory found for pattern '{dir_pattern}' in {self.saveDir}.")
                dirs.sort(reverse=True)  # Most recent first
                results_dir = os.path.join(self.saveDir, dirs[0])
            else:
                results_dir = os.path.join(self.saveDir, f'results_{results_date}_{opt_name}')
                if optimizer == OptimizerType.MLEM:
                    pass
                elif optimizer == OptimizerType.LS:
                    results_dir += f'_Alpha_{self.alpha}'
                if not os.path.exists(results_dir):
                    raise FileNotFoundError(f"Directory {results_dir} does not exist.")
            # Load reconstruction results
            recon_key = 'reconPhantom' if withTumor else 'reconLaser'
            recon_path = os.path.join(results_dir, f'{recon_key}.npy')
            if not os.path.exists(recon_path):
                raise FileNotFoundError(f"No reconstruction file found at {recon_path}.")
            data = np.load(recon_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.ndim == 3:
                if withTumor:
                    self.reconPhantom = [data[i, :, :] for i in range(data.shape[0])]
                else:
                    self.reconLaser = [data[i, :, :] for i in range(data.shape[0])]
            else:
                if withTumor:
                    self.reconPhantom = data
                else:
                    self.reconLaser = data
            # Load saved indices as list of 2D arrays
            indices_path = os.path.join(results_dir, 'indices.npy')
            if not os.path.exists(indices_path):
                raise FileNotFoundError(f"No indices file found at {indices_path}.")
            indices_data = np.load(indices_path, allow_pickle=True)
            if isinstance(indices_data, np.ndarray) and indices_data.ndim == 3:
                self.indices = [indices_data[i, :, :] for i in range(indices_data.shape[0])]
            else:
                self.indices = indices_data
            if show_logs:
                print(f"Loaded reconstruction results and indices from {results_dir}")
        
    def normalizeSMatrix(self):
        self.SMatrix = self.SMatrix / (float(self.experiment.params.acoustic['emission']['voltage'])*float(self.experiment.params.acoustic['emission']['sensitivity']))  

    # PRIVATE METHODS
         
    def _fillSMatrix(self, isShowLogs=True):
        if self.smatrixType == SMatrixType.DENSE:
            return self._fillSMatrix_DENSE(isShowLogs=isShowLogs)
        elif self.smatrixType == SMatrixType.CSR:
            return self._fillSMatrix_CSR(isShowLogs=isShowLogs)
        elif self.smatrixType == SMatrixType.COO:
            raise NotImplementedError("COO sparse matrix not implemented yet.")
        elif self.smatrixType == SMatrixType.SELL:
            return self._fillSMatrix_SELL(isShowLogs=isShowLogs)
        else:
            raise ValueError(f"Unsupported SMatrix type: {self.smatrixType}")
    
    def _fillSMatrix_DENSE(self, isShowLogs=True):
        """
        Build a dense matrix using SMatrix_DENSE class.
        Frees all temporary memory at each step.
        """
        dense_matrix = SMatrix_DENSE(self.experiment, device=self.device)
        dense_matrix.allocate()
        if isShowLogs:
            print(f" Dense matrix size: {dense_matrix.getMatrixSize()} GB")
        return dense_matrix
    
    def _fillSMatrix_CSR(self, isShowLogs=True):
        """
        Built a sparse CSR matrix in chunks without intermediate concatenation.
        Frees all temporary memory at each step.
        """
        sparse_matrix = SMatrix_CSR(self.experiment,relative_threshold=self.sparseThreshold,device=self.device)
        sparse_matrix.allocate()
        if isShowLogs:
            print(f" Sparse matrix size: {sparse_matrix.getMatrixSize()} GB")
            print(f"Sparse matrix density: {sparse_matrix.compute_density()}")
        return sparse_matrix
    
    def _fillSMatrix_SELL(self, isShowLogs=True):
        """
        Built a sparse SELL matrix in chunks without intermediate concatenation.
        Frees all temporary memory at each step.
        """
        sparse_matrix = SMatrix_SELL(self.experiment,relative_threshold=self.sparseThreshold,device=self.device)
        sparse_matrix.allocate()
        if isShowLogs:
            print(f" Sparse matrix size: {sparse_matrix.getMatrixSize()} GB")
            print(f"Sparse matrix density: {sparse_matrix.compute_density()}")
        return sparse_matrix
        

    
    def flip_angle(self):
        if self.smatrixType == SMatrixType.CSR:
            self.SMatrix.flip_angle()
    
    # STATIC METHODS
    @staticmethod
    def plot_mse_comparison(recon_list, figSize=(4.5, 3.5), labels=None):
        """
        Affiche les courbes de MSE pour chaque reconstruction dans recon_list.

        Args:
            recon_list (list): Liste d'objets recon (doivent avoir les attributs 'indices' et 'MSE').
            figSize (tuple, optional): Taille de la figure.
            labels (list, optional): Liste des labels pour chaque courbe. Si None, utilise "Recon i".
        """
        if labels is None:
            labels = [f"Recon {i+1}" for i in range(len(recon_list))]

        plt.figure(figsize=figSize)
        colors = ['red', 'green', 'blue', 'orange', 'purple']  # Ajoute d'autres couleurs si nécessaire

        for i, recon in enumerate(recon_list):
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else f"Recon {i+1}"

            # Trouve l'index et la valeur minimale du MSE
            best_idx = recon.indices[np.argmin(recon.MSE)]
            min_mse = np.min(recon.MSE)

            # Trace la courbe de MSE
            plt.plot(recon.indices, recon.MSE, f'{color}-', label=label)
            # Ligne horizontale pour le min MSE
            plt.axhline(min_mse, color=color, linestyle='--', alpha=0.5)
            # Ligne verticale pour l'itération du min MSE
            plt.axvline(best_idx, color=color, linestyle='--', alpha=0.5)

        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title("MSE vs. Iteration (Comparison)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-")

        # Légende personnalisée
        handles = []
        for i, recon in enumerate(recon_list):
            color = colors[i % len(colors)]
            best_idx = recon.indices[np.argmin(recon.MSE)]
            min_mse = np.min(recon.MSE)
            handles.append(
                plt.Line2D([0], [0], color=color,
                        label=f"{labels[i] if labels and i < len(labels) else f'Recon {i+1}'} (min={min_mse:.4f} @ it.{best_idx+1})")
            )

        plt.legend(handles=handles, loc='upper right')
        plt.tight_layout()
        plt.show()

