import concurrent
import warnings
from AOT_biomaps.Config import config

from ._mainRecon import Recon
from .ReconEnums import ReconType, OptimizerType, ProcessType, SMatrixType, PotentialType, PotentialShapeType, StopCriterionType
from .AOT_Preconditioner.PreconditionerEnums import PreconditionerType
from .AOT_Preconditioner import DiagPreconditioner, NoPreconditioner
from .AOT_Optimizers import MLEM, PGD, MAPEM, DEPIERRO, PDHG, PGC, PPGMLEM, LBFGS, FISTA
from .AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from .AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from .AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

import os
import subprocess
import numpy as np
from datetime import datetime
from tempfile import gettempdir
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, List
from IPython.display import HTML

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# ============================================================================
# ALGORITHM FORMULAS (for error messages and documentation)
# ============================================================================

ALGORITHM_FORMULAS = {
    OptimizerType.MLEM: {
        "formula": "λ^(k+1) = λ^(k) * (A^T * (y / (A*λ^(k) + ε))) / (A^T * 1)",
        "description": "Maximum Likelihood Expectation Maximization (multiplicative form)",
        "reference": "Shepp and Vardi, IEEE TMI, 1982",
        "required_params": ["denominatorThreshold"],
        "constraints": {
            "numIterations": "> 0",
            "denominatorThreshold": "> 0",
        },
        "notes": "Native Poisson solver. Unregularized, tends to amplify high-frequency noise at high iterations.",
        "potentialFunction": [PotentialType.NONE]
    },
    OptimizerType.PGD: {
        "formula": "λ^(k+1) = [ λ^(k) - α * M^-1 * A^T * (A*λ^(k) - y) ]_+",
        "description": "Projected Gradient Descent (Least-Squares minimization with non-negativity constraint)",
        "reference": "Landweber, 1951",
        "required_params": ["alpha"],
        "constraints": {
            "alpha": "> 0 or 'auto'",
            "numIterations": "> 0",
            "numIterations_stepCalculation": "> 0",
        },
        "notes": "Gaussian noise solver. Alpha='auto' uses power method for Lipschitz estimation. Hardly recommanded to set eta to 1",
        "potentialFunction": [PotentialType.NONE]
    },
    OptimizerType.MAPEM: {
        "formula": "λ^(k+1) = [ λ^(k) * (A^T * (y / (A*λ^(k) + ε))) / (A^T * 1 + ∇U) ]_+",
        "description": "Maximum A Posteriori Expectation Maximization (One-Step Late)",
        "reference": "Green, IEEE TMI, 1990",
        "required_params": ["beta", "delta"],
        "constraints": {
            "beta": ">= 0",
            "delta": ">= 0",
            "numIterations": "> 0",
        },
        "notes": "Structurally unstable for large beta due to gradient in denominator. Use DEPIERRO instead.",
        "potentialFunction": [PotentialType.QUADRATIC, PotentialType.HUBER, PotentialType.RELATIVE_DIFFERENCE]
    },
    OptimizerType.DEPIERRO: {
        "formula": "λ^(k+1) = [ λ^(k) + λ^(k) * (∇EM - ∇U) / (A^T * 1 + λ^(k) * H_U) ]_+",
        "description": "De Pierro's Optimization Transfer (Separable Paraboloidal Surrogate)",
        "reference": "De Pierro, IEEE TMI, 1995",
        "required_params": ["beta", "delta"],
        "constraints": {
            "beta": ">= 0",
            "delta": "> 0",
            "numIterations": "> 0",
        },
        "notes": "Monotonically convergent and highly stable surrogate method for Poisson MRF penalties.",
        "potentialFunction": [PotentialType.QUADRATIC, PotentialType.HUBER, PotentialType.RELATIVE_DIFFERENCE]
    },
    OptimizerType.PPGMLEM: {
        "formula": "λ^(k+1) = [ λ^(k) + α * (∇EM - ∇U) / (A^T * 1 + δ * H_U + γ) ]_+",
        "description": "Penalized Preconditioned Gradient ML-EM",
        "reference": "Nuyts et al., IEEE TNS, 2002",
        "required_params": ["beta", "delta"],
        "constraints": {
            "beta": ">= 0",
            "delta": ">= 0",
            "numIterations": "> 0",
        },
        "notes": "Additive Poisson gradient descent stabilized by pseudo-Hessian and Tikhonov parameter.",
        "potentialFunction": [PotentialType.QUADRATIC, PotentialType.HUBER, PotentialType.RELATIVE_DIFFERENCE]
    },
    OptimizerType.PGC: {
        "formula": "d_k = -∇f + β_cg * d_{k-1} | λ^(k+1) = [ λ^(k) + α * d_k ]_+",
        "description": "Penalized Gauss-Newton Conjugate Gradient (Polak-Ribière)",
        "reference": "Standard Nonlinear Conjugate Gradient",
        "required_params": ["alpha", "beta", "delta"],
        "constraints": {
            "alpha": "> 0 or 'auto'",
            "beta": ">= 0",
            "delta": ">= 0",
            "numIterations": "> 0",
            "numIterations_stepCalculation": "> 0",
        },
        "notes": "Highly accelerated convergence for Least-Squares (Gaussian) fidelity.",
        "potentialFunction": [PotentialType.QUADRATIC, PotentialType.HUBER, PotentialType.RELATIVE_DIFFERENCE]
    },
    OptimizerType.PDHG: {
        "formula": "x = prox_{τ*TV}(x - τ*A^Ty) | y = y + σ*(A(2x - x_{old}) - data)",
        "description": "Primal-Dual Hybrid Gradient (Chambolle-Pock)",
        "reference": "Chambolle and Pock, J. Math. Imaging Vis., 2011",
        "required_params": ["beta", "gamma", "theta", "tau", "sigma"],
        "constraints": {
            "beta": ">= 0",
            "gamma": "> 0",
            "theta": ">= 0",
            "tau": "> 0 or 'auto'",
            "sigma": "> 0 or 'auto'",
            "numIterations": "> 0",
            "numSubsets": ">= 1",
            "reshufflePeriod": ">= 0",
        },
        "notes": "Mathematically robust solver for non-differentiable Total Variation (L1).",
        "potentialFunction": [PotentialType.TOTAL_VARIATION]
    },
    OptimizerType.LBFGS: {
        "formula": "w^(k+1) = w^(k) - step * H_k * ∇f(w^(k)) | λ = w^2",
        "description": "Limited-memory BFGS quasi-Newton optimization (Unconstrained Variable Transform)",
        "reference": "Liu and Nocedal, Mathematical Programming, 1989",
        "required_params": ["beta", "delta"],
        "constraints": {
            "beta": ">= 0",
            "delta": "> 0",
            "numIterations": "> 0",
        },
        "notes": "Uses λ = w^2 transform to inherently enforce non-negativity without projection artifacts.",
        "potentialFunction": [PotentialType.QUADRATIC, PotentialType.HUBER, PotentialType.RELATIVE_DIFFERENCE]
    },
    OptimizerType.FISTA: {
        "formula": "z = x + ((t_k - 1) / t_{k+1}) * (x - x_old) | x = prox_{α*U}(z - α*∇f(z))",
        "description": "Fast Iterative Shrinkage-Thresholding Algorithm (Accelerated Proximal Gradient)",
        "reference": "Beck and Teboulle, SIAM J. Imaging Sci., 2009",
        "required_params": ["alpha", "beta", "delta", "eta"],
        "constraints": {
            "alpha": "> 0 or 'auto'",
            "beta": ">= 0",
            "delta": ">= 0",
            "eta": "> 0",
            "numIterations": "> 0",
            "numIterations_stepCalculation": "> 0",
        },
        "notes": "Accelerated proximal gradient method with Nesterov momentum. Requires proximal operator for the potential function.",
        "potentialFunction": [PotentialType.QUADRATIC, PotentialType.HUBER, PotentialType.RELATIVE_DIFFERENCE]
    },
}

class AlgebraicRecon(Recon):
    """
    Algebraic reconstruction class for AOT_biomaps.
    
    This class provides a unified interface for all iterative reconstruction algorithms,
    including MLEM, PGD, MAPEM, DEPIERRO, PPGMLEM, PGC, PDHG, and FISTA.
    
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
            potentialFunction=PotentialType.HUBER
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
        numIterations: int = 100,
        numSubsets: int = 1,
        isSavingEachIteration: bool = True,
        isCostFunction: bool = False,
        maxSaves: int = 5000,
        denominatorThreshold: float = 1e-6,
        # Sparsing
        smatrixType: SMatrixType = SMatrixType.SELL,
        sparseThreshold: float = 0.1,
        blockRows: int = 64,
        sliceHeight: int = 64,
        sigma_sell: int = 4096,
        isComplexRecon: bool = False,
        device: Optional[str] = None,
        # Preconditioning
        preconditionerType: Optional[PreconditionerType] = PreconditionerType.NONE,
        # Regularization parameters
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        # Parameters for automatic step size calculation
        eta: Optional[float] = None,
        numIterations_stepCalculation: Optional[int] = 20,
        # PDHG-specific parameters
        theta: Optional[float] = None,
        tau: Optional[float] = None,
        sigma: Optional[float] = None,
        reshufflePeriod: Optional[int] = None,
        # Potential function parameters
        PotentialShape: Optional[PotentialShapeType] = PotentialShapeType.CROSS,
        PotentialRadius: Optional[int] = 2,
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
            isCostFunction: Whether to compute cost function (default: False)
            maxSaves: Maximum number of intermediate saves (default: 5000)
            denominatorThreshold: Threshold for denominator to avoid division by zero (default: 1e-6)
            smatrixType: Type of system matrix (default: SMatrixType.SELL)
            sparseThreshold: Threshold for sparse matrix construction (default: 0.1)
            blockRows: Number of rows per block for sparse matrix construction (default: 64) (only used for CSR and SELL)
            sliceHeight: Number of rows per slice for SELL format (default: 64)
            isComplexRecon: Whether to perform complex reconstruction (default: False)
            device: Device to use ('cpu' or 'gpu') (default: auto-detected)
            preconditionerType: Type of preconditioner (PreconditionerType.NONE or DIAGONAL, default: NONE)
            alpha: Step size for PGD, FISTA (default: None)
            beta: Regularization parameter for MAPEM, DEPIERRO, PPGMLEM, PGC, PDHG (default: None)
            gamma: Preconditioning parameter for PPGMLEM (default: None)
            delta: Huber threshold or relative difference parameter for MAPEM, PPGMLEM, DEPIERRO (default: None)
            eta: Parameter for the Lipschitz constant estimation (must be < 2 for convergence and > 1 for faster convergence). Useless if alpha is a float. (default: None)
            numIterations_stepCalculation: Number of iterations for automatic step size calculation (default: 20). Used for power method estimation of the Lipschitz constant when alpha is set to "auto".
            theta: Extrapolation parameter for PDHG (default: None)
            tau: Primal step size for PDHG (default: None)
            sigma: Dual step size for PDHG (default: None)
            PotentialShape: Shape parameter for potential functions (default: PotentialShapeType.CROSS). Useless for TOTAL_VARIATION potential which use cross shape by default.
            PotentialRadius: Radius parameter for potential functions (default: 2). Useless for TOTAL_VARIATION potential which use radius = 1 by default.
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
            if optimizer in (OptimizerType.PPGMLEM, OptimizerType.PGC, OptimizerType.DEPIERRO):
                self.potentialFunction = PotentialType.HUBER
            else:
                self.potentialFunction = None
        
        # Store common parameters
        self.numIterations = numIterations
        self.numSubsets = numSubsets
        self.isSavingEachIteration = isSavingEachIteration
        self.isCostFunction = isCostFunction
        self.maxSaves = maxSaves
        self.denominatorThreshold = denominatorThreshold
        self.isComplexRecon = isComplexRecon

        if device is None:
            self.device = f'gpu:{config.select_best_gpu()}' if CUPY_AVAILABLE else 'cpu'
        else:
            if type(device) is not str:
                print(f"[AOT-biomaps] Error occurred while setting device. Must be 'cpu' or 'gpu:<index>'. Falling back to auto-detection.")
                self.device = f'gpu:{config.select_best_gpu()}' if CUPY_AVAILABLE else 'cpu'
            elif device not in ['cpu'] and not device.startswith('gpu:'):
                print(f"[AOT-biomaps] Error occurred while setting device. Must be 'cpu' or 'gpu:<index>'. Falling back to auto-detection.")
                self.device = f'gpu:{config.select_best_gpu()}' if CUPY_AVAILABLE else 'cpu'
            else:
                self.device = device

        self.SMatrix = None
        self.smatrixType = smatrixType
        self.sparseThreshold = sparseThreshold
        self.blockRows = blockRows
        self.sliceHeight = sliceHeight
        self.sigma_sell = sigma_sell
        self.preconditionerType = preconditionerType

        # Store regularization parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma

        # Store parameters for automatic step size calculation
        self.eta = eta
        self.numIterations_stepCalculation = numIterations_stepCalculation

        # Store PDHG-specific parameters
        self.theta = theta
        self.tau = tau
        self.sigma = sigma
        self.reshufflePeriod = reshufflePeriod
        
        # Set corner and face with defaults
        self.PotentialShape = PotentialShape
        self.PotentialRadius = PotentialRadius
        
        # Initialize reconstruction results
        self.reconPhantom: List[np.ndarray] = []
        self.reconLaser: List[np.ndarray] = []
        self.cost_historyPhantom = None
        self.cost_historyLaser = None
        self.indices: List[int] = []
        self.MSE: Optional[List[float]] = None
        self.SSIM: Optional[List[float]] = None
        self.CRC: Optional[List[float]] = None
               
        # Handle complex reconstruction
        if self.isComplexRecon:
            if self.experiment.AOsignal_withTumor is not None:
                self.experiment.AOsignal_withTumor_demodulated = self.experiment.demodulate_AOsignal(withTumor=True)
            elif self.experiment.AOsignal_withoutTumor is not None:
                self.experiment.AOsignal_withoutTumor_demodulated = self.experiment.demodulate_AOsignal(withTumor=False)
            else:
                raise ValueError("[AOT-biomaps] No AO signal available for demodulation. Please provide at least one signal, with or without tumor.")
            self.experiment.AcousticFields_demodulated = self.experiment.demodulate_acoustic_fields()
    
    def _validate_potential_compatibility(self, errors: list):
        """
        Validate that the selected potential function is compatible with the optimizer,
        and that the geometrical parameters (shape, radius) are mathematically sound.
        """
        POTENTIAL_COMPATIBILITY = {
            PotentialType.QUADRATIC: [
                OptimizerType.MAPEM, OptimizerType.DEPIERRO, OptimizerType.PPGMLEM,
                OptimizerType.PGC, OptimizerType.PDHG, OptimizerType.LBFGS, OptimizerType.FISTA
            ],
            PotentialType.HUBER: [
                OptimizerType.MAPEM, OptimizerType.DEPIERRO, OptimizerType.PPGMLEM, OptimizerType.PGC,
                OptimizerType.PDHG, OptimizerType.LBFGS, OptimizerType.FISTA
            ],
            PotentialType.RELATIVE_DIFFERENCE: [
                OptimizerType.MAPEM, OptimizerType.DEPIERRO, OptimizerType.PPGMLEM, OptimizerType.PGC,
                OptimizerType.PDHG, OptimizerType.LBFGS, OptimizerType.FISTA
            ],
            PotentialType.TOTAL_VARIATION: [
                OptimizerType.PDHG
            ],
            PotentialType.NONE: [
                OptimizerType.MLEM, OptimizerType.PGD, OptimizerType.MAPEM, OptimizerType.DEPIERRO, 
                OptimizerType.PPGMLEM, OptimizerType.PGC, OptimizerType.PDHG, OptimizerType.LBFGS, OptimizerType.FISTA
            ],
        }
        
        current_potential = self.potentialFunction if self.potentialFunction is not None else PotentialType.NONE

        if current_potential not in POTENTIAL_COMPATIBILITY:
            errors.append(f"[AOT-biomaps] Unknown potential function: {current_potential}")
            return
        
        compatible_optimizers = POTENTIAL_COMPATIBILITY[current_potential]
        if self.optimizer not in compatible_optimizers:
            compatible_names = [opt.value for opt in compatible_optimizers]
            errors.append(
                f"[AOT-biomaps] Potential '{current_potential.value}' is not compatible with optimizer '{self.optimizer.value}'. "
                f"[AOT-biomaps] Compatible optimizers: {', '.join(compatible_names)}"
            )
        
        # Hyperparameters dependency checks per potential type
        if current_potential == PotentialType.TOTAL_VARIATION:
            if self.beta is None:
                errors.append("[AOT-biomaps] TOTAL_VARIATION potential requires 'beta' parameter to be set.")
            if self.PotentialShape != PotentialShapeType.CROSS or self.PotentialRadius != 1:
                errors.append(f"[AOT-biomaps] TOTAL_VARIATION strictly requires shape=CROSS and radius=1 for proximal evaluation. Got shape={self.PotentialShape}, radius={self.PotentialRadius}.")
        
        elif current_potential == PotentialType.HUBER:
            if self.delta is None:
                errors.append("[AOT-biomaps] HUBER potential requires 'delta' parameter to be set.")
        
        elif current_potential == PotentialType.RELATIVE_DIFFERENCE:
            if self.beta is None:
                errors.append("[AOT-biomaps] RELATIVE_DIFFERENCE potential requires 'beta' parameter to be set.")
            if self.delta is None:
                errors.append("[AOT-biomaps] RELATIVE_DIFFERENCE potential requires 'delta' parameter to be set.")

    def _validate_hyperparameters(self):
        """Validate all hyperparameters and stopping criteria for the selected optimizer."""
        if self.optimizer not in ALGORITHM_FORMULAS:
            warnings.warn(f"[AOT-biomaps] Unknown optimizer type: {self.optimizer}. Skipping hyperparameter validation.")
            return

        formula_info = ALGORITHM_FORMULAS[self.optimizer]
        errors = []

        # 1. Structural checks
        if self.numIterations <= 0:
            errors.append(f"[AOT-biomaps] numIterations must be > 0, got {self.numIterations}")
        if self.numSubsets <= 0:
            errors.append(f"[AOT-biomaps] numSubsets must be > 0, got {self.numSubsets}")

        # 2. Validate Stopping Criteria Logic
        stop_crit = getattr(self, 'stop_criterion', StopCriterionType.MAX_ITERATIONS)
        if stop_crit != StopCriterionType.MAX_ITERATIONS:
            threshold = getattr(self, 'stop_threshold', None)
            if threshold is None or threshold <= 0:
                errors.append(f"[AOT-biomaps] Stopping criterion {stop_crit.name} requires a positive 'stop_threshold'. Got {threshold}.")
            if stop_crit == StopCriterionType.MSE:
                # Basic check to ensure we are in a simulated context if MSE is requested
                if self.experiment.OpticImage is None or self.experiment.OpticImage.phantom is None:
                    errors.append("[AOT-biomaps] MSE stopping criterion requires a simulated Ground Truth (phantom) in the experiment.")

        # 3. Check optimizer-specific mathematical constraints
        constraints = formula_info.get("constraints", {})
        constraints_display = formula_info.get("constraints_display", {})

        for param_name, constraint in constraints.items():
            param_value = getattr(self, param_name, None)
            display_name = constraints_display.get(param_name, param_name)

            if param_value is None:
                if param_name in formula_info.get("required_params", []):
                    errors.append(f"[AOT-biomaps] Required hyperparameter '{display_name}' is not set.")
                continue

            if constraint == "> 0 or 'auto'":
                if not (param_value == 'auto' or (isinstance(param_value, (int, float)) and param_value > 0)):
                    errors.append(f"[AOT-biomaps] '{display_name}' must be > 0 or 'auto', got '{param_value}'")

            elif constraint == "> 0":
                if not (isinstance(param_value, (int, float)) and param_value > 0):
                    errors.append(f"[AOT-biomaps] '{display_name}' must be strictly > 0, got '{param_value}'")

            elif constraint == ">= 0":
                if not (isinstance(param_value, (int, float)) and param_value >= 0):
                    errors.append(f"[AOT-biomaps] '{display_name}' must be >= 0, got '{param_value}'")

            elif constraint.startswith("in ["):
                try:
                    low, high = map(float, constraint[4:-1].split(","))
                    if not (low <= param_value <= high):
                        errors.append(f"[AOT-biomaps] '{display_name}' must be in closed interval [{low}, {high}], got '{param_value}'")
                except ValueError:
                    pass

        # 4. Check global cross-compatibility
        self._validate_potential_compatibility(errors)

        # 5. Format and raise validation errors stack
        if errors:
            error_msg = f"\n{'='*80}\n"
            error_msg += f"HYPERPARAMETER VALIDATION ERROR FOR METRIC: {self.optimizer.name}\n"
            error_msg += f"{'='*80}\n\n"
            error_msg += f"Algorithm Context : {formula_info['description']}\n"
            error_msg += f"Literature Reference: {formula_info.get('reference', 'N/A')}\n"
            error_msg += f"Governing Formula   : {formula_info['formula']}\n\n"
            
            error_msg += "Required Execution Parameters:\n"
            for param_name in formula_info.get("required_params", []):
                param_value = getattr(self, param_name, None)
                param_display = constraints_display.get(param_name, param_name)
                param_constraint = constraints.get(param_name, "N/A")
                error_msg += f"  - {param_display}: requirement [{param_constraint}] (currently populated: {param_value})\n"
                
            error_msg += f"\nActive Architectural Bounds:\n"
            for param_name, constraint in constraints.items():
                param_value = getattr(self, param_name, None)
                param_display = constraints_display.get(param_name, param_name)
                error_msg += f"  - {param_display}: standard bounds [{constraint}] (currently populated: {param_value})\n"
                
            if formula_info.get("notes"):
                error_msg += f"\nAlgorithm Notes:\n  {formula_info['notes']}\n"
                
            error_msg += f"\nRaised Exceptions Stack:\n"
            for error in errors:
                error_msg += f"  [Constraint Violation] -> {error}\n"
            error_msg += f"\n{'='*80}\n"
            raise ValueError(error_msg)
        
    # PUBLIC METHODS
    def generate_SMatrix(self, isShowLogs=True):
        if self.smatrixType == SMatrixType.DENSE:
            self.SMatrix = self._fill_SMatrix_DENSE(isShowLogs=isShowLogs)
        elif self.smatrixType == SMatrixType.CSR:
            self.SMatrix = self._fill_SMatrix_CSR(isShowLogs=isShowLogs)
        elif self.smatrixType == SMatrixType.COO:
            raise NotImplementedError(f"[AOT-biomaps] COO sparse matrix not implemented yet.")
        elif self.smatrixType == SMatrixType.SELL:
            self.SMatrix = self._fill_SMatrix_SELL(isShowLogs=isShowLogs)
        else:
            raise ValueError(f"[AOT-biomaps] Unsupported SMatrix type: {self.smatrixType}")
    
    def flip_probe(self):
        self.SMatrix.flip_probe()

    def apply_apodization(self, window_vector: np.ndarray):
        self.SMatrix.apply_apodization(window_vector)
    
    def run(self, y = None, processType: ProcessType = ProcessType.PYTHON, withTumor: bool = True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs: bool = True):
        """
        Run the algebraic reconstruction process.
        
        Dispatches to the appropriate reconstruction method based on processType.
        
        Args:
            y: The observed data (if provided). else, it will be loaded from the experiment based on withTumor flag.
            processType: Type of processing (PYTHON or CASToR)
            withTumor: If True, reconstruct with tumor data; otherwise without
            stop_criterion: Criterion for stopping the reconstruction
            stop_threshold: Threshold for the stopping criterion
            show_criterion: If True, display the stopping criterion
            show_logs: If True, display progress logs
            
        Raises:
            ValueError: If SMatrix is not generated or processType is unknown
        """
        if self.SMatrix is None:
            raise ValueError("System matrix (SMatrix) is not generated. Please call generate_SMatrix() before run().")
        
        if processType == ProcessType.CASToR:
            self._algebraic_recon_CASToR(withTumor=withTumor, show_logs=show_logs)
        elif processType == ProcessType.PYTHON:
            self._algebraic_recon_Python(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        else:
            raise ValueError(f"[AOT-biomaps] Unknown Algebraic reconstruction type: {processType}")

    def _algebraic_recon_Python(self, y=None, withTumor: bool = True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs: bool = True):
        """
        Run algebraic reconstruction using Python implementation.
        
        Dispatches to the appropriate optimizer-specific method based on self.optimizer.
        
        Args:
            y: The observed data (if provided). else, it will be loaded from the experiment based on withTumor flag.
            withTumor: If True, reconstruct with tumor data; otherwise without
            stop_criterion: Criterion for stopping the reconstruction
            stop_threshold: Threshold for the stopping criterion
            stop_window_size: Window size (used to avoid early stop due to oscillations)
            show_criterion: If True, display the stopping criterion
            show_logs: If True, display progress logs
            
        Raises:
            ValueError: If the optimizer is not supported
        """
        # Check signal availability
        if y is None:
            if withTumor:
                if self.experiment.AOsignal_withTumor is None:
                    raise ValueError("[AOT-biomaps] AO signal with tumor is not available. Please generate AO signal with tumor in the experiment first.")
                y = self.experiment.AOsignal_withTumor if not self.isComplexRecon else np.array([self.experiment.AOsignal_withTumor_demodulated[key] for key in self.experiment.AOsignal_withTumor_demodulated.keys()]).T
            else:
                if self.experiment.AOsignal_withoutTumor is None:
                    raise ValueError("[AOT-biomaps] AO signal without tumor is not available. Please generate AO signal without tumor in the experiment first.")
                y = self.experiment.AOsignal_withoutTumor if not self.isComplexRecon else np.array([self.experiment.AOsignal_withoutTumor_demodulated[key] for key in self.experiment.AOsignal_withoutTumor_demodulated.keys()]).T

        self._validate_hyperparameters()

        # Dispatch to optimizer-specific method
        if self.optimizer == OptimizerType.MLEM:
            self._run_MLEM(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        elif self.optimizer == OptimizerType.PGD:
            self._run_PGD(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        elif self.optimizer == OptimizerType.MAPEM:
            self._run_MAPEM(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        elif self.optimizer == OptimizerType.DEPIERRO:
            self._run_DEPIERRO(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        elif self.optimizer == OptimizerType.PPGMLEM:
            self._run_PPGMLEM(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        elif self.optimizer == OptimizerType.FISTA:
            self._run_FISTA(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        elif self.optimizer == OptimizerType.PGC:
            self._run_PGC(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        elif self.optimizer == OptimizerType.PDHG:
            self._run_PDHG(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        elif self.optimizer == OptimizerType.LBFGS:
            self._run_LBFGS(y=y, withTumor=withTumor, stop_criterion=stop_criterion, stop_threshold=stop_threshold, stop_window_size=stop_window_size, show_criterion=show_criterion, show_logs=show_logs)
        else:
            raise ValueError(f"[AOT-biomaps] Unsupported optimizer type: {self.optimizer}")

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
                print(f"[AOT-biomaps] Missing .cdh file. Generating {fileName}...")
            self.experiment.saveAOsignals_Castor(self.saveDir)

        # Check/generate system matrix
        if not os.path.isdir(smatrix):
            os.makedirs(smatrix, exist_ok=True)
        if not os.listdir(smatrix):
            if show_logs:
                print(f"[AOT-biomaps] System matrix missing. Generating...")
            self.experiment.saveAcousticFields(self.saveDir)

        # Verify that the .cdh file exists
        if not os.path.isfile(os.path.join(self.saveDir, fileName)):
            raise FileNotFoundError(f"[AOT-biomaps] .cdh file does not exist: {fileName}")

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
        elif self.optimizer == OptimizerType.PGD:
            if self.alpha is not None:
                cmd.extend(["-alpha", str(self.alpha)])
        elif self.optimizer == OptimizerType.MAPEM:
            if self.alpha is not None:
                cmd.extend(["-alpha", str(self.alpha)])
            if self.beta is not None:
                cmd.extend(["-beta", str(self.beta)])
        elif self.optimizer == OptimizerType.DEPIERRO:
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
            print(f"[AOT-biomaps] Running reconstruction with CASToR...")
        result = subprocess.run(recon_script_path, env=env, check=True, capture_output=True, text=True)

        # Display CASToR output for debugging
        if show_logs:
            print("CASToR output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)

        if show_logs:
            print(f"[AOT-biomaps] Reconstruction completed successfully.")
        self.load_reconCASToR(withTumor=withTumor)

    def _run_MLEM(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run MLEM reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = MLEM(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                denominator_threshold=self.denominatorThreshold,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = MLEM(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                denominator_threshold=self.denominatorThreshold,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )

    def _run_PGD(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run Projected Gradient Descent reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = PGD(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                alpha=self.alpha,
                numIterations_stepCalculation=self.numIterations_stepCalculation,
                preconditioner_type=self.preconditionerType,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = PGD(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                alpha=self.alpha,
                numIterations_stepCalculation=self.numIterations_stepCalculation,
                preconditioner_type=self.preconditionerType,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )

    def _run_LBFGS(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run LBFGS reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = LBFGS(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                delta=self.delta,   
                potential_type=self.potentialFunction,  
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = LBFGS(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                delta=self.delta,   
                potential_type=self.potentialFunction,  
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )

    def _run_MAPEM(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run MAPEM reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = MAPEM(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                delta=self.delta,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = MAPEM(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                delta=self.delta,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )

    def _run_DEPIERRO(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run DEPIERRO reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = DEPIERRO(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                delta=self.delta,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = DEPIERRO(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                delta=self.delta,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction = self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )

    def _run_PPGMLEM(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run PPGMLEM reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = PPGMLEM(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                delta=self.delta,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction=self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = PPGMLEM(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                delta=self.delta,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction=self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )

    def _run_FISTA(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run FISTA reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = FISTA(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                alpha=self.alpha,
                beta=self.beta,
                delta=self.delta,
                numIterations_stepCalculation=self.numIterations_stepCalculation,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                preconditioner_type=self.preconditionerType,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction=self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = FISTA(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                alpha=self.alpha,
                beta=self.beta,
                delta=self.delta,
                numIterations_stepCalculation=self.numIterations_stepCalculation,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                preconditioner_type=self.preconditionerType,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction=self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )

    def _run_PGC(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run PGC reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = PGC(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                alpha=self.alpha,
                beta=self.beta,
                delta=self.delta,
                eta=self.eta,
                numIterations_stepCalculation=self.numIterations_stepCalculation,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction=self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = PGC(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                alpha=self.alpha,
                beta=self.beta,
                delta=self.delta,
                eta=self.eta,
                numIterations_stepCalculation=self.numIterations_stepCalculation,
                potential_type=self.potentialFunction,
                potential_shape=self.PotentialShape,
                potential_radius=self.PotentialRadius,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction=self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )

    def _run_PDHG(self, y, withTumor=True, stop_criterion=StopCriterionType.MAX_ITERATIONS, stop_threshold=None, stop_window_size=1, show_criterion=True, show_logs=True):
        """Run PDHG reconstruction."""
        if withTumor:
            self.reconPhantom, self.indices, self.cost_historyPhantom = PDHG(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                theta=self.theta,
                tau=self.tau,
                sigma=self.sigma,
                numIterations_stepCalculation=self.numIterations_stepCalculation,
                num_subsets=self.numSubsets,
                reshuffle_period=self.reshufflePeriod,
                preconditioner_type=self.preconditionerType,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction=self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
        else:
            self.reconLaser, self.indices, self.cost_historyLaser = PDHG(
                SMatrix=self.SMatrix,
                y=y,
                numIterations=self.numIterations,
                beta=self.beta,
                theta=self.theta,
                tau=self.tau,
                sigma=self.sigma,
                numIterations_stepCalculation=self.numIterations_stepCalculation,
                num_subsets=self.numSubsets,
                reshuffle_period=self.reshufflePeriod,
                preconditioner_type=self.preconditionerType,
                stop_criterion=stop_criterion,
                stop_threshold=stop_threshold,
                stop_window_size=stop_window_size,
                isSavingEachIteration=self.isSavingEachIteration,
                isCostFunction=self.isCostFunction,
                withTumor=withTumor,
                max_saves=self.maxSaves,
                show_logs=show_logs,
                show_criterion=show_criterion
            )
    
    def plot_cost(self, isSaving=True, log_scale_x=False, log_scale_y=False, figSize=(4,3), show_logs=True):
        """
        Plot the cost function values.

        Parameters:
            isSaving: bool, whether to save the plot.
            log_scale_x: bool, if True, use logarithmic scale for the x-axis.
            log_scale_y: bool, if True, use logarithmic scale for the y-axis.
        Returns:
            None
        """
        if self.cost_historyPhantom is None and self.cost_historyLaser is None:
            raise ValueError("[AOT-biomaps] Cost function history is empty. Please calculate it first.")
        if self.cost_historyPhantom is not None and len(self.cost_historyPhantom) < 1 or self.cost_historyLaser is not None and len(self.cost_historyLaser) < 1:
            raise ValueError("[AOT-biomaps] Plotting cost function requires more than one data point. Please set isSavingEachIteration=True and isCostFunction=True when running the reconstruction to plot cost function history.")

        # Plot cost function curve
        plt.figure(figsize=figSize)
        if self.cost_historyPhantom is not None:
            plt.plot(self.indices, self.cost_historyPhantom/np.max(self.cost_historyPhantom), 'r-', label="Cost (Phantom)")
        if self.cost_historyLaser is not None:
            plt.plot(self.indices, self.cost_historyLaser/np.max(self.cost_historyLaser), 'b-', label="Cost (Laser)")


        plt.xlabel("Iteration")
        plt.ylabel("Normalized Cost")
        plt.title("Cost Function vs. Iteration")
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
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_Cost_plot_{self.optimizer.name}_{scale_str}{date_str}.png')
            plt.savefig(SavingFolder, dpi=300)
            if show_logs:
                print(f"[AOT-biomaps] Cost plot saved to {SavingFolder}")
        plt.show()
    
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
            raise ValueError("[AOT-biomaps] MSE is empty. Please calculate MSE first.")
        if self.MSE is not None and len(self.MSE) < 1:
            raise ValueError("[AOT-biomaps] Plotting MSE function requires more than one data point. Please set isSavingEachIteration=True and isCostFunction=True when running the reconstruction to plot MSE history.")  


        best_idx = self.indices[np.argmin(self.MSE)]
        if show_logs:
            print(f"[AOT-biomaps] Lowest MSE = {np.min(self.MSE):.4f} at iteration {best_idx+1}")
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
                print(f"[AOT-biomaps] MSE plot saved to {SavingFolder}")

        plt.show()

    def show_MSE_bestRecon(self, isSaving=True, show_logs=True, figSize=(15, 5)):
        if not self.MSE:
            raise ValueError("[AOT-biomaps] MSE is empty. Please calculate MSE first.")

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
                print(f"[AOT-biomaps] MSE plot saved to {SavingFolder}")

        plt.show()

    def show_lambda_animation(self, vmin=None, vmax=None, total_duration_ms=3000, save_path=None, max_frames=1000, figSize=(4, 4), isPropMSE=True, show_logs=True):
        """
        Show lambda iteration animation with speed proportional to MSE acceleration.
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
            raise ValueError("[AOT-biomaps] Not enough lambda matrices available for animation.")

        if isPropMSE and (self.MSE is None or len(self.MSE) == 0):
            raise ValueError("[AOT-biomaps] MSE is empty or not calculated. Please calculate MSE first.")

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
            raise ValueError("[AOT-biomaps] SSIM is empty. Please calculate SSIM first.")

        best_idx = self.indices[np.argmax(self.SSIM)]
        if show_logs:
            print(f"[AOT-biomaps] Highest SSIM = {np.max(self.SSIM):.4f} at iteration {best_idx+1}")
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
                print(f"[AOT-biomaps] SSIM plot saved to {SavingFolder}")

        plt.show()

    def show_SSIM_bestRecon(self, isSaving=True, figSize=(15, 5), show_logs=True):
        
        if not self.SSIM:
            raise ValueError("[AOT-biomaps] SSIM is empty. Please calculate SSIM first.")

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
                print(f"[AOT-biomaps] SSIM plot saved to {SavingFolder}")
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
            raise ValueError("[AOT-biomaps] Reconstructed laser is empty. Run reconstruction first.")
        if isinstance(self.reconLaser, list) and len(self.reconLaser) == 1:
            raise ValueError("[AOT-biomaps] Reconstructed Image without tumor is a single frame. Run with isSavingEachIteration=True.")
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")
        if isinstance(self.reconPhantom, list) and len(self.reconPhantom) == 1:
            raise ValueError("[AOT-biomaps] Reconstructed Image with tumor is a single frame. Run with isSavingEachIteration=True.")

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
                print("[AOT-biomaps] Warning: saveDir is None. Configure saving path to save the figure.")
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
                    print(f"[AOT-biomaps] Plot saved to: {save_path}")

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
                raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")
            if isinstance(self.reconPhantom, list) and len(self.reconPhantom) == 1:
                raise ValueError("[AOT-biomaps] Reconstructed Image with tumor is a single frame. Run reconstruction with isSavingEachIteration=True.")
            recon_list = self.reconPhantom
            ground_truth = self.experiment.OpticImage.phantom
            title_suffix = "with_tumor"
        else:
            if self.reconLaser is None or self.reconLaser == []:
                raise ValueError("[AOT-biomaps] Reconstructed laser is empty. Run reconstruction first.")
            if isinstance(self.reconLaser, list) and len(self.reconLaser) == 1:
                raise ValueError("[AOT-biomaps] Reconstructed Image without tumor is a single frame. Run reconstruction with isSavingEachIteration=True.")
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
                print(f"[AOT-biomaps] Figure saved to: {save_path}")

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
            raise ValueError("[AOT-biomaps] Save directory is not specified.")
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
        Load reconstruction results (reconPhantom or reconLaser) and indices.
        If results_date is None, finds the most recent directory matching the pattern.
        """
        recon_key = 'reconPhantom' if withTumor else 'reconLaser'

        if filePath is not None:
            # Direct load mode from a specified file
            if not os.path.exists(filePath):
                raise FileNotFoundError(f"[AOT-biomaps] No reconstruction file found at {filePath}.")
            recon_path = filePath
        else:
            if self.saveDir is None:
                raise ValueError("[AOT-biomaps] Save directory is not specified. Please set saveDir before loading.")

            # Determine the optimizer name to use
            opt_name = optimizer.value if optimizer is not None else self.optimizer.value

            # Build the base directory pattern (e.g., "results_*_PDHG")
            dir_pattern = f'results_*_{opt_name}'

            # Add optimizer-specific parameters to the pattern
            if optimizer is None:
                optimizer = self.optimizer
            if optimizer == OptimizerType.PPGMLEM:
                dir_pattern += f'_Beta_{self.beta}_Delta_{self.delta}_Gamma_{self.gamma}_Sigma_{self.sigma}'
            elif optimizer in (OptimizerType.PGC, OptimizerType.DEPIERRO):
                dir_pattern += f'_Beta_{self.beta}_Sigma_{self.sigma}'
            elif optimizer == OptimizerType.PGD:
                dir_pattern += f'_Alpha_{self.alpha}'

            # List all directories in self.saveDir
            all_dirs = [d for d in os.listdir(self.saveDir) if os.path.isdir(os.path.join(self.saveDir, d))]

            # Filter directories matching the pattern (e.g., "results_0906_PDHG")
            matching_dirs = []
            for d in all_dirs:
                if d.startswith('results_') and f'_{opt_name}' in d:
                    matching_dirs.append(d)

            if not matching_dirs:
                raise FileNotFoundError(f"[AOT-biomaps] No matching results directory found for pattern 'results_*_{opt_name}' in {self.saveDir}.")

            # If results_date is specified, use it
            if results_date is not None:
                target_dir = f'results_{results_date}_{opt_name}'
                if optimizer == OptimizerType.PPGMLEM:
                    target_dir += f'_Beta_{self.beta}_Delta_{self.delta}_Gamma_{self.gamma}_Sigma_{self.sigma}'
                elif optimizer in (OptimizerType.PGC, OptimizerType.DEPIERRO):
                    target_dir += f'_Beta_{self.beta}_Sigma_{self.sigma}'
                elif optimizer == OptimizerType.PGD:
                    target_dir += f'_Alpha_{self.alpha}'

                # Check if the directory exists
                results_dir = os.path.join(self.saveDir, target_dir)
                if not os.path.exists(results_dir):
                    raise FileNotFoundError(f"[AOT-biomaps] Directory {results_dir} does not exist.")
            else:
                # Find the most recent directory (sorted by date in ddmm format)
                matching_dirs.sort(reverse=True)  # Sort alphabetically (ddmm dates are sortable)
                results_dir = os.path.join(self.saveDir, matching_dirs[0])

            # Path to the reconstruction file
            recon_path = os.path.join(results_dir, f'{recon_key}.npy')
            if not os.path.exists(recon_path):
                raise FileNotFoundError(f"[AOT-biomaps] No {recon_key}.npy file found in {results_dir}.")

        # Load the file (3D array or list of 2D arrays)
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

        # Load indices if they exist
        indices_path = os.path.join(os.path.dirname(recon_path), 'indices.npy')
        if os.path.exists(indices_path):
            indices_data = np.load(indices_path, allow_pickle=True)
            if isinstance(indices_data, np.ndarray) and indices_data.ndim == 3:
                self.indices = [indices_data[i, :, :] for i in range(indices_data.shape[0])]
            else:
                self.indices = indices_data
        else:
            self.indices = None

        if show_logs:
            print(f"[AOT-biomaps] Loaded reconstruction results and indices from {recon_path}")
        
    # PRIVATE METHODS
             
    def _fill_SMatrix_DENSE(self, isShowLogs=True):
        """
        Build a real or complex dense matrix using SMatrix_DENSE class.
        Frees all temporary memory at each step.
        """
        print("[AOT-biomaps] Building DENSE SMatrix") if isShowLogs else None
        SMatrix = SMatrix_DENSE(experiment=self.experiment, device=self.device, isComplexSMatrix=self.isComplexRecon)
        SMatrix.allocate()
        SMatrix.normalize_matrix()
        if isShowLogs:
            print(f"[AOT-biomaps] DENSE SMatrix size: {SMatrix.get_matrix_size()['total_gb']:.2f} GB")
        return SMatrix
    
    def _fill_SMatrix_CSR(self, isShowLogs=True):
        """
        Built a real or complex sparse CSR matrix in chunks without intermediate concatenation.
        Frees all temporary memory at each step.
        """
        print("[AOT-biomaps] Building CSR SMatrix with relative threshold =", self.sparseThreshold) if isShowLogs else None
        SMatrix = SMatrix_CSR(experiment=self.experiment, device=self.device, block_rows=self.blockRows, relative_threshold=self.sparseThreshold, isComplexSMatrix=self.isComplexRecon)
        SMatrix.allocate()
        SMatrix.normalize_matrix()
        if isShowLogs:
            print(f"[AOT-biomaps] CSR SMatrix size: {SMatrix.get_matrix_size()['total_gb']:.2f} GB")
            print(f"[AOT-biomaps] CSR sparse matrix density: {SMatrix.compute_density():.2f}%")
        return SMatrix
    
    def _fill_SMatrix_SELL(self, isShowLogs=True):
        """
        Built a real or complex sparse SELL matrix in chunks without intermediate concatenation.
        Frees all temporary memory at each step.
        """
        print("[AOT-biomaps] Building SELL SMatrix with relative threshold =", self.sparseThreshold) if isShowLogs else None
        SMatrix = SMatrix_SELL(experiment=self.experiment, device=self.device, block_rows=self.blockRows, relative_threshold=self.sparseThreshold, slice_height=self.sliceHeight, sigma=self.sigma_sell, isComplexSMatrix=self.isComplexRecon)
        SMatrix.allocate()
        SMatrix.normalize_matrix()
        if isShowLogs:
            print(f"[AOT-biomaps] SELL SMatrix size: {SMatrix.get_matrix_size()['total_gb']:.2f} GB")
            print(f"[AOT-biomaps] SELL sparse matrix density: {SMatrix.compute_density():.2f}%")
        return SMatrix
        
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
        colors = ['red', 'green', 'blue', 'orange', 'purple'] 

        for i, recon in enumerate(recon_list):
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else f"Recon {i+1}"


            best_idx = recon.indices[np.argmin(recon.MSE)]
            min_mse = np.min(recon.MSE)


            plt.plot(recon.indices, recon.MSE, f'{color}-', label=label)
            plt.axhline(min_mse, color=color, linestyle='--', alpha=0.5)
            plt.axvline(best_idx, color=color, linestyle='--', alpha=0.5)

        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title("MSE vs. Iteration (Comparison)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-")

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

    def show(self, withTumor=True, savePath=None, scale='same', figsize=(8, 4)):
        """
        Display the reconstructed images with a properly positioned colorbar.
        Args:
            withTumor (bool): If True, displays reconPhantom. If False, displays reconLaser. Default is True.
            savePath (str): Path to save the figure. If None, the figure is not saved. Default is None.
            scale (str): Scale for the aspect ratio of the plots. Default is 'same'. Options are 'same' or 'auto'.
            figsize (tuple): Figure size (width, height). Default is (8, 4).

        Note:
            Requires matplotlib to be installed. If matplotlib is not available, this method will raise an ImportError.
        """
        extent = [self.experiment.params.general['Xrange'][0] * 1e3, self.experiment.params.general['Xrange'][1] * 1e3, self.experiment.params.general['Zrange'][1] * 1e3, self.experiment.params.general['Zrange'][0] * 1e3]

        # Determine the image to display
        if withTumor:
            if self.reconPhantom is None:
                raise ValueError("[AOT-biomaps] Reconstructed phantom with tumor is empty. Run reconstruction first.")
            if isinstance(self.reconPhantom, (list, tuple)) and len(self.reconPhantom) == 0:
                raise ValueError("[AOT-biomaps] Reconstructed phantom with tumor is empty. Run reconstruction first.")
            image = self.reconPhantom[-1] if isinstance(self.reconPhantom, list) else self.reconPhantom
            ground_truth = self.experiment.OpticImage.phantom if self.experiment.OpticImage else None
            title_recon = "Reconstructed phantom with tumor"
            title_gt = "Phantom with tumor"
        else:
            if self.reconLaser is None:
                raise ValueError("[AOT-biomaps] Reconstructed laser without tumor is empty. Run reconstruction first.")
            if isinstance(self.reconLaser, (list, tuple)) and len(self.reconLaser) == 0:
                raise ValueError("[AOT-biomaps] Reconstructed laser without tumor is empty. Run reconstruction first.")
            image = self.reconLaser[-1] if isinstance(self.reconLaser, list) else self.reconLaser
            ground_truth = self.experiment.OpticImage.laser.intensity if self.experiment.OpticImage else None
            title_recon = "Reconstructed laser without tumor"
            title_gt = "Laser without tumor"

        # Gestion propre des sous-graphes avec squeeze=False pour garantir un tableau 2D
        n_cols = 2 if ground_truth is not None else 1
        fig, axs = plt.subplots(1, n_cols, figsize=figsize if n_cols == 2 else (figsize[0]/2, figsize[1]), squeeze=False)

        if ground_truth is not None:
            vmin, vmax = (0, 1) if scale == 'same' else (np.min(image), np.max(image))
        else:
            vmin, vmax = (0, np.max(image))

        im0 = axs[0, 0].imshow(image, cmap='hot', vmin=vmin, vmax=vmax, extent=extent, aspect='equal')
        axs[0, 0].set_title(title_recon)
        axs[0, 0].set_xlabel("X (mm)")
        axs[0, 0].set_ylabel("Z (mm)")
        axs[0, 0].tick_params(axis='both', which='major')

        # Plot ground truth if available
        if ground_truth is not None:
            gt_vmin, gt_vmax = (0, 1) if scale == 'same' else (np.min(ground_truth), np.max(ground_truth))

            im1 = axs[0, 1].imshow(ground_truth, cmap='hot', vmin=gt_vmin, vmax=gt_vmax, extent=extent, aspect='equal')
            axs[0, 1].set_title(title_gt)
            axs[0, 1].set_xlabel("X (mm)")
            axs[0, 1].set_ylabel("Z (mm)")
            axs[0, 1].tick_params(axis='both', which='major')

        plt.subplots_adjust(bottom=0.15, wspace=0.3)

        # Calculate colorbar position dynamically based on figsize
        cbar_width = 0.05 * figsize[0] / figsize[1]  # Relative to figure height
        cbar_height = 0.05
        cbar_x = 0.25  # Centered horizontally
        cbar_y = -0.06 # Positioned at the bottom

        # Add colorbar
        cbar_ax = fig.add_axes([cbar_x, cbar_y, 0.5, cbar_height])
        cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
        if ground_truth is not None and scale == 'same':
            cbar.set_label('Normalized Intensity') 
        else:
            cbar.set_label('Intensity')
        cbar.ax.tick_params(labelsize=8)

        # Save figure if path is provided
        if savePath is not None:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            filename = 'recon_with_tumor.png' if withTumor else 'recon_without_tumor.png'
            plt.savefig(os.path.join(savePath, filename), dpi=300, bbox_inches='tight')

        plt.show()