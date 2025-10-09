# Non-Negative Least Squares Algorithm with Diagonal Preconditioning

This document provides a complete mathematical description of a **non-negative least squares (NNLS) algorithm** with diagonal preconditioning, optimized for GPU implementation. The algorithm combines projected gradient descent with preconditioning to accelerate convergence.

---

## **1. Problem Statement**

We aim to solve the following constrained optimization problem:

$$
\min_{\lambda \geq 0} \frac{1}{2} \| y - A \lambda \|_2^2
$$

where:
- $A \in \mathbb{R}^{TN \times ZX}$ is a data matrix (also known as the design matrix or system matrix),
- $y \in \mathbb{R}^{TN}$ is an observation vector,
- $\lambda \in \mathbb{R}^{ZX}$ is the vector of unknowns, subject to the non-negativity constraint $\lambda \geq 0$.

$$
\lambda^{(k+1)}=\lambda^{(k)}-\alpha \nabla J(\lambda^{(k)})
$$

Where the objective function $J$ is defined as half the squared norm:

$$
J(\lambda^{(k)}) = \frac{1}{2} \| y - A \lambda^{(k)} \|_2^2 
$$

and its associated derived functions is :

$$
\nabla J(\lambda^{(k)}) = -A^T \(y -A \lambda^{(k)} \) 
$$

Thus,


$$
\textcolor{red}{\lambda^{(k+1)} = \lambda^{(k)} + \alpha A^T (y - A \lambda^{(k)})}
$$

-------

## **2. Demonstration**

The squared $L_2$-norm of the difference between $y$ and $A \lambda$ is defined as:

$$
\| y - A \lambda \|_2^2 = \sum_{i=1}^{TN} (y_i - (A \lambda)_i)^2
$$

where $(A \lambda)_i$ is the $i$-th component of the vector $A \lambda$.
The objective function $J$ is defined as half the squared norm:

$$
J(\lambda) = \frac{1}{2} \| y - A \lambda \|_2^2 = \frac{1}{2} \sum_{i=1}^{TN} (y_i - (A \lambda)_i)^2
$$

The factor $\frac{1}{2}$ is introduced to simplify subsequent derivative calculations.
The squared norm can be expressed in matrix form using the dot product:

$$
\| y - A \lambda \|_2^2 = (y - A \lambda)^T (y - A \lambda)
$$

Thus, the objective function $J$ becomes:

$$
J(\lambda) = \frac{1}{2} (y - A \lambda)^T (y - A \lambda)
$$

Let's expand this expression:

$$
J(\lambda) = \frac{1}{2} \left( y^T y - 2 y^T A \lambda + \lambda^T A^T A \lambda \right)
$$

$$
(y - A \lambda)^T (y - A \lambda) = y^T y - y^T A \lambda - \lambda^T A^T y + \lambda^T A^T A \lambda
$$

- $y^T A \lambda$ is a scalar, so $y^T A \lambda = (y^T A \lambda^{(k)})^T = \lambda^T A^T y$.
- Therefore, $y^T A \lambda + \lambda^T A^T y = 2 y^T A \lambda$.

$$
J(\lambda) = \frac{1}{2} y^T y - y^T A \lambda + \frac{1}{2} \lambda^T A^T A \lambda
$$

To solve this optimization problem, it is useful to compute the gradient of $J$ with respect to $\lambda$.

The gradient of $J$ is given by:

$$
\nabla J(\lambda) = -A^T y + A^T A \lambda
$$

$$
\nabla J(\lambda) = A^T (y - A \lambda)
$$

Or,

$$
\nabla J(\lambda) = A^T r
$$

with the residual $r= (y - A \lambda)$


---

## **2. Data Normalization**

To improve numerical stability, the input data is normalized as follows:

$$
A_{\text{flat}} \leftarrow \frac{A_{\text{flat}}}{\max(A_{\text{flat}}) + \epsilon}, \quad y_{\text{flat}} \leftarrow \frac{y_{\text{flat}}}{\max(y_{\text{flat}}) + \epsilon}
$$

where $\epsilon = 10^{-8}$ is a small constant to prevent division by zero.

---

## **3. Initialization**

The algorithm is initialized with:

$$
\lambda^{(0)} = 0
$$

---

## **4. Diagonal Preconditioning**

A diagonal preconditioner is computed to accelerate convergence:

$$
M = \text{diag}(A_{\text{flat}}^T A_{\text{flat}})
$$

The inverse of the preconditioner is given by:

$$
M^{-1} = \frac{1}{\max(\text{diag}(A_{\text{flat}}^T A_{\text{flat}}), 10^{-6})}
$$

where $\max(\cdot, 10^{-6})$ ensures that diagonal elements are not too small.

---

## **5. Iterative Algorithm**

The algorithm iteratively updates $\lambda$ using a projected gradient method with a fixed step size. At each iteration $k$, the following steps are performed:

### **a. Residual Calculation**

The residual $r^{(k)}$ is computed as:

$$
r^{(k)} = y_{\text{flat}} - A_{\text{flat}} \lambda^{(k)}
$$

### **b. Preconditioned Gradient Calculation**

The gradient of the objective function is:

$$
\nabla f(\lambda^{(k)}) = -A_{\text{flat}}^T r^{(k)}
$$

The gradient is preconditioned by multiplying with $M^{-1}$:

$$
g^{(k)} = M^{-1} \nabla f(\lambda^{(k)}) = -M^{-1} A_{\text{flat}}^T r^{(k)}
$$

### **c. Variable Update**

The variables are updated with a fixed step size $\alpha$:

$$
\lambda^{(k+1)} = \lambda^{(k)} + \alpha g^{(k)}
$$

where $\alpha$ is a fixed step size (e.g., $\alpha = 10^{-3}$).

### **d. Projection onto \(\mathbb{R}^+\)**

To satisfy the non-negativity constraint, $\lambda^{(k+1)}$ is projected onto $\mathbb{R}^+$:

$$
\lambda^{(k+1)} \leftarrow \max(0, \lambda^{(k+1)})
$$

---
## **6. Denormalization**

At the end of the algorithm, the solution is denormalized to restore the original scale of the data:

$$
\lambda_{\text{final}} = \lambda^{(K)} \cdot \frac{\max(y)}{\max(A)}
$$

----

## **6. Iteration Saving Strategy**

To avoid memory explosion, iterations are saved intelligently:

- If the total number of iterations $K \leq 5000$, all iterations are saved.
- If $K > 5000$, only iterations $k$ such that $k \mod n = 0$ are saved, where $n = \lfloor K / 5000 \rfloor$.

The indices of saved iterations are also recorded for later reference.

---


## **12. References**

- [[1] Hestenes, M. R., & Stiefel, E. (1952).
*Methods of Conjugate Gradients for Solving Linear Systems*.
National Bureau of Standards Report **2951**.
US Government Printing Office.](https://www.stat.uchicago.edu/~lekheng/courses/302/classics/hestenes-stiefel.pdf)

