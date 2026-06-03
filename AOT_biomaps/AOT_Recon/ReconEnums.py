from enum import Enum

class ReconType(Enum):
    """
    Enum for different reconstruction types.

    Selection of reconstruction types:
    - Analytic: A reconstruction method based on analytical solutions.
    - Algebraic: A reconstruction method using algebraic techniques.
    - Algebraic: A reconstruction method that Algebraicly refines the solution.
    - Bayesian: A reconstruction method based on Bayesian statistical approaches.
    - DeepLearning: A reconstruction method utilizing deep learning algorithms.
    """

    Analytic = 'analytic'
    """A reconstruction method based on analytical solutions."""
    Algebraic = 'algebraic'
    """A reconstruction method that Algebraicly refines the solution."""
    Bayesian = 'bayesian'
    """A reconstruction method based on Bayesian statistical approaches."""
    DeepLearning = 'deep_learning'
    """A reconstruction method utilizing deep learning algorithms."""
    Convex = 'convex'

class AnalyticType(Enum):
    iFOURIER = 'iFOURIER'
    """
    This analytic reconstruction type uses the inverse Fourier transform to reconstruct the image.
    It is suitable for data that can be represented in the frequency domain.
    It is typically used for data that has been transformed into the frequency domain, such as in Fourier optics.
    It is not suitable for data that has not been transformed into the frequency domain.
    """
    iRADON = 'iRADON'
    """
    This analytic reconstruction type uses the inverse Radon transform to reconstruct the image.
    It is suitable for data that has been transformed into the Radon domain, such as in computed tomography (CT).
    It is typically used for data that has been transformed into the Radon domain, such as in CT.
    It is not suitable for data that has not been transformed into the Radon domain.
    """

class OptimizerType(Enum):
    """
    Enum for optimization algorithms used in reconstruction.
    
    Available optimizers and their properties:
    - MLEM: Maximum Likelihood Expectation Maximization (multiplicative form)
    - LS: Landweber (Least Squares) algorithm
    - MAPEM: Maximum A Posteriori Expectation Maximization
    - DEPIERRO: De Pierro's optimization transfer algorithm
    - PPGMLEM: Penalized Preconditioned Gradient MLEM
    - PGC: Penalized Gauss-Newton Conjugate Gradient
    - PDHG: Primal-Dual Hybrid Gradient
    """
    MLEM = 'MLEM'
    """
    Maximum Likelihood Expectation Maximization.
    Multiplicative form implementation that truncates negative data to 0.
    Supports subsets (becomes OSEM).
    Compatible with: emission and transmission data, histogram and list-mode.
    """
    LS = 'LS'
    """
    Landweber Least Squares algorithm.
    Uses log-converted model for transmission data.
    Requires manual relaxation parameter tuning.
    Compatible with: histogram data, emission and transmission.
    """
    MAPEM = 'MAPEM'
    """
    Maximum A Posteriori Expectation Maximization.
    Gradient-based algorithm for penalized ML reconstruction.
    Compatible with: histogram data, emission only.
    """
    DEPIERRO = 'DEPIERRO'
    """
    De Pierro's optimization transfer algorithm (1995).
    Convergent algorithm for ML reconstruction with MRF penalty.
    Numerically robust to high penalty strength.
    Compatible with: histogram and list-mode data, emission only.
    """
    PPGMLEM = 'PPGMLEM'
    """
    Penalized Preconditioned Gradient MLEM.
    Heuristic gradient ascent algorithm for penalized ML reconstruction.
    Addresses numerical issues of OSL with large penalty strengths.
    Compatible with: histogram data, emission only.
    """
    PGC = 'PGC'
    """
    Penalized Gauss-Newton Conjugate Gradient.
    For penalized ML reconstruction with second-order derivative potentials.
    Compatible with: histogram data, emission only.
    """
    PDHG = 'PDHG'
    """
    Primal-Dual Hybrid Gradient.
    For non-differentiable potentials like Total Variation.
    Compatible with: all data types.
    """
    LBFGS = 'LBFGS'
    """
    Limited-memory BFGS.
    Quasi-Newton optimization algorithm with regularization support.
    Compatible with: differentiable potential functions (QUADRATIC, HUBER, RELATIVE_DIFFERENCE).
    """
    PIGD = 'PIGD'
    """
    Penalized Iterative Gradient Descent (PIGD).
    Gradient-based optimization algorithm for penalized ML reconstruction. 
    Compatible with: differentiable potential functions (QUADRATIC, HUBER, RELATIVE_DIFFERENCE).
    """

class PotentialType(Enum):
    """
    Enum for potential functions used in regularization.
    
    All potential functions penalize differences between neighboring voxels:
    p(u, v) = p(u - v)
    
    Compatibility with optimizers:
    - QUADRATIC: Compatible with all optimizers (MLEM, LS, MAPEM, DEPIERRO, PPGMLEM, PGC, PDHG)
    - HUBER: Compatible with MAPEM, PPGMLEM, PGC, PDHG (differentiable)
    - RELATIVE_DIFFERENCE: Compatible with MAPEM, PPGMLEM, PGC (differentiable)
    - TOTAL_VARIATION: Compatible with PDHG only (non-differentiable, not suitable for gradient-based methods)
    
    Note: TOTAL_VARIATION is non-differentiable at zero and returns a subgradient.
    It is NOT compatible with MLEM, LS, MAPEM, DEPIERRO, PPGMLEM, PGC, or other gradient-based optimizers.
    """
    NONE = 'NONE'
    """
    No regularization (potential is zero).
    """

    QUADRATIC = 'QUADRATIC'
    """
    Quadratic potential: p(u, v) = 0.5 * alpha * (u - v)^2
    
    Properties:
    - Differentiable: Yes
    - Convex: Yes
    - Compatible optimizers: All (MLEM, LS, MAPEM, DEPIERRO, PPGMLEM, PGC, PDHG)
    
    Reference: Geman and Geman, IEEE Trans. Pattern Anal. Machine Intell., vol. PAMI-6, pp. 721-741, 1984.
    """

    HUBER = 'HUBER'
    """
    Huber piecewise potential:
    p(u, v, delta) = 
        - 0.5 * (u - v)^2, if |u - v| <= delta
        - delta * |u - v| - 0.5 * delta^2, otherwise
    
    Properties:
    - Differentiable: Yes (with continuous derivative at delta)
    - Convex: Yes
    - Compatible optimizers: MAPEM, PPGMLEM, PGC, PDHG
    - Not compatible: MLEM, LS, DEPIERRO
    
    Parameters:
    - delta: Threshold for switching between quadratic and linear behavior (default: 0.01)
    
    Reference: Mumcuoglu et al, Phys. Med. Biol., vol. 41, pp. 1777-1807, 1996.
    """

    RELATIVE_DIFFERENCE = 'RELATIVE_DIFFERENCE'
    """
    Relative difference potential (Nuyts):
    p(u, v, beta) = alpha * (u - v)^2 / (u + v + beta * |u - v|)
    
    Properties:
    - Differentiable: Yes (where u + v + beta * |u - v| > 0)
    - Convex: No (edge-preserving, non-convex)
    - Compatible optimizers: MAPEM, PPGMLEM, PGC
    - Not compatible: MLEM, LS, DEPIERRO, PDHG
    
    Parameters:
    - beta: Regularization parameter for the denominator (default: 1.0)
    
    Reference: Nuyts et al, IEEE Trans. Nucl. Sci., vol. 49, pp. 56-60, 2002.
    """

    TOTAL_VARIATION = 'TOTAL_VARIATION'
    """
    Total Variation potential (anisotropic):
    p(u, v) = alpha * |u - v|
    
    Properties:
    - Differentiable: No (non-differentiable at zero, returns subgradient)
    - Convex: Yes
    - Compatible optimizers: PDHG only
    - Not compatible: MLEM, LS, MAPEM, DEPIERRO, PPGMLEM, PGC
    
    Note: TV regularization preserves edges while reducing noise.
    It requires primal-dual methods like PDHG that can handle non-differentiable functions.
    
    Reference: Chambolle and Pock, J. Math. Imaging Vis., 2011.
    """

class PotentialShapeType(Enum):
    CROSS = 'CROSS'
    """Penalizes differences between the center voxel and its n face-connected neighbors."""
    SQUARE = 'SQUARE'
    """Penalizes differences between the center voxel and its n face-connected neighbors, as well as between the face-connected neighbors themselves (4-neighborhood in 2D, 6-neighborhood in 3D)."""
    CIRCLE = 'CIRCLE'
    """Penalizes differences between the center voxel and all neighbors within a specified radius, as well as between those neighbors themselves (8-neighborhood in 2D, 26-neighborhood in 3D)."""

class ProcessType(Enum):
    CASToR = 'CASToR'
    PYTHON = 'PYTHON'

class NoiseType(Enum):
    """
    Enum for different noise types used in reconstructions.
    
    Selection of noise types:
    - Poisson: Poisson noise, typically used for emission data.
    - Gaussian: Gaussian noise, typically used for transmission data.
    - None: No noise is applied.
    """
    POISSON = 'poisson'
    """Poisson noise."""
    GAUSSIAN = 'gaussian'
    """Gaussian noise."""
    None_ = 'none'
    """No noise is applied."""

class PreconditionerType(Enum):
    """
    Enum for preconditioning types used in iterative reconstruction.
    
    Available preconditioners:
    - NONE: No preconditioning (identity)
    - DIAGONAL: Diagonal preconditioning using inverse of diagonal elements
    """
    NONE = 'NONE'
    """No preconditioning applied."""
    DIAGONAL = 'DIAGONAL'
    """Diagonal preconditioning: M^-1 where M is diagonal matrix with A^T*1."""

class SMatrixType(Enum):
    """
    Enum for different sparsing methods used in reconstructions.
    
    Selection of sparsing methods:
    - Thresholding: Sparsing based on a threshold value.
    - TopK: Sparsing by retaining the top K values.
    - None: No sparsing is applied.
    """
    DENSE = 'DENSE'
    """No sparsing is applied."""
    CSR = 'CSR'
    """Sparsing based on a threshold value."""
    COO = 'COO'
    """Sparsing by retaining the top K values."""
    SELL = 'SELL'
    """Sparsing using sell C sigma method.
    Optimized variant of ELLPACK, dividing the matrix into fixed-size "chunks" of `C` rows.
    Non-zero elements are sorted by column within each chunk to improve memory coalescing on GPUs.
    Rows are padded with zeros to align their length to the longest row in the chunk.
    ** Ref : Kreutzer, M., Hager, G., Wellein, G., Fehske, H., & Bishop, A. R. (2014).
          "A Unified Sparse Matrix Data Format for Efficient General Sparse Matrix-Vector Multiply on Modern Processors".
          ACM Transactions on Mathematical Software, 41(2), 1–24. DOI: 10.1145/2592376.
    """

class StopCriterionType(Enum):
    """
    Enum for different stopping criteria used in iterative reconstruction algorithms.
    
    Selection of stopping criteria:
    - MaxIterations: Stop after a maximum number of iterations.
    - CostFunction: Stop when the cost function value falls below a threshold.
    - RelativeChange: Stop when the relative change in the solution is below a threshold.
    - GradientNorm: Stop when the norm of the gradient is below a threshold.
    """
    MAX_ITERATIONS = 'MAX_ITERATIONS'
    """Stop after a maximum number of iterations."""
    COST_FUNCTION = 'COST_FUNCTION'
    """Stop when the cost function value falls below a threshold."""
    RELATIVE_CHANGE = 'RELATIVE_CHANGE'
    """Stop when the relative change in the solution is below a threshold."""
    GRADIENT_NORM = 'GRADIENT_NORM'
    """Stop when the norm of the gradient is below a threshold."""