# Maximum Likelihood Expectation Maximization (MLEM) Algorithm

This document provides a complete mathematical description of the **MLEM algorithm**. MLEM is an iterative method for reconstructing images from incomplete or noisy projections, commonly used in tomographic imaging.

---
## **1. Problem Statement**

MLEM aims to solve the following optimization problem:

$$
\max_{\lambda \geq 0} \mathcal{L}(\lambda)
$$

where $\mathcal{L}(\lambda)$ is the log-likelihood function for Poisson-distributed data:

$$
\mathcal{L}(\lambda) = \sum_{i=1}^{TN} y_i \log((A \lambda)_i) - (A \lambda)_i
$$

where:
- $A \in \mathbb{R}^{TN \times ZX}$ is the system matrix (also known as the projection matrix),
- $y \in \mathbb{R}^{TN}$ is the observation vector (measured sinogram),
- $\lambda \in \mathbb{R}^{ZX}$ is the vector of unknowns (image to reconstruct), subject to the non-negativity constraint $\lambda \geq 0$.

---
## **2. Derivation of the MLEM Update Rule**

The MLEM algorithm iteratively updates the estimate of $\lambda$ using the following update rule:

$$
\lambda_j^{(k+1)} = \frac{\lambda_j^{(k)}}{\sum_{i=1}^{TN} A_{ji}} \sum_{i=1}^{TN} A_{ji} \frac{y_i}{(A \lambda^{(k)})_i}
$$

### **a. Log-Likelihood Function**

The log-likelihood function for Poisson noise is given by:

$$
\mathcal{L}(\lambda) = \sum_{i=1}^{TN} y_i \log((A \lambda)_i) - (A \lambda)_i
$$

### **b. Gradient of the Log-Likelihood**

The gradient of $\mathcal{L}(\lambda)$ with respect to $\lambda_j$ is:

$$
\frac{\partial \mathcal{L}}{\partial \lambda_j} = \sum_{i=1}^{TN} A_{ji} \left( \frac{y_i}{(A \lambda)_i} - 1 \right)
$$

### **c. MLEM Update Rule**

The MLEM update rule is derived by setting the gradient to zero and solving iteratively:

$$
\lambda_j^{(k+1)} = \frac{\lambda_j^{(k)}}{S_j} \sum_{i=1}^{TN} A_{ji} \frac{y_i}{(A \lambda^{(k)})_i}
$$

In matrix form, this can be written as:

$$
\lambda^{(k+1)} = \lambda^{(k)} \circ \frac{A^T e^{(k)}}{S + \epsilon}
$$

where:
- $S \in \mathbb{R}^{ZX}$ is the sensitivity vector, with components $S_j = \sum_{i=1}^{TN} A_{ji}$,
- $e^{(k)} = \frac{y}{A \lambda^{(k)} + \epsilon}$ is the correction factor,
- $\circ$ denotes element-wise multiplication,
- $\epsilon$ is a small constant to prevent division by zero.

---
## **3. Data Preparation**

The input data is prepared as follows:
- The system matrix $A$ is reshaped into a 2D matrix $A_{flat} \in \mathbb{R}^{TN \times ZX}$,
- The observation vector $y$ is flattened into $y_{flat} \in \mathbb{R}^{TN}$,
- The sensitivity vector $S$ is precomputed as:

$$
S_j = \sum_{i=1}^{TN} A_{ji}, \quad \forall j \in \{1, \dots, ZX\}
$$

or in matrix form:

$$
S = A^T \mathbf{1}
$$

where $\mathbf{1}$ is a vector of ones of size $TN$.

---
## **4. Initialization**

The algorithm is initialized with a uniform image:

$$
\lambda^{(0)} = \mathbf{1}
$$

---
## **5. Iterative Algorithm**

The algorithm iteratively updates $\lambda$ using the MLEM update rule. At each iteration $k$, the following steps are performed:

### **a. Forward Projection**

Compute the forward projection:

$$
q^{(k)} = A_{flat} \lambda^{(k)}
$$

### **b. Correction Factor**

Compute the correction factor:

$$
e^{(k)} = \frac{y_{flat}}{q^{(k)} + \epsilon}
$$

where $\epsilon$ is a small constant to prevent division by zero.

### **c. Back-Projection**

Compute the back-projection of the correction factor:

$$
c^{(k)} = A_{flat}^T e^{(k)}
$$

### **d. Update Step**

Update $\lambda$ using the MLEM update rule:

$$
\lambda^{(k+1)} = \lambda^{(k)} \circ \frac{c^{(k)}}{S + \epsilon}
$$

---
## **7. Iteration Saving Strategy**

To avoid memory explosion, iterations are saved intelligently:
- If the total number of iterations $K \leq 1000$, all iterations are saved,
- If $K > 1000$, only iterations $k$ such that $k \mod n = 0$ are saved, where $n = \lfloor K / 1000 \rfloor$.

The indices of saved iterations are also recorded for later reference.
