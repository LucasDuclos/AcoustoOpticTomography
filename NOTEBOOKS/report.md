
 <span style="color:red; font-weight:bold; text-decoration:underline;">Acousto-Optic</span>

# Introduction

## Formation of an Acousto-Optic Signal

As its name suggests, acousto-optic is a bimodal imaging technique that combines acoustics and optics to reconstruct an optical image deep within a scattering medium. This is not possible with standard optical imaging modalities.

![schéma_tomo_AVEC_particules2](https://github.com/user-attachments/assets/b8ebaed4-4c7a-4dea-80f0-039e9902b7ea)

### Optical Image $\lambda$

In a scattering medium, incident light deviates from its path when it encounters particles. Beyond a certain transport length, the incident photon loses memory of its initial direction. The transport length is given by:

$$
l^{*} = \frac{l_s}{1 - g}
$$

where $l_s$ is the mean distance between scattering events, and $g$ defines the directionality of the light. When $g = 1$, light propagates in a straight line. In the case of biological tissues, $g \approx 0.8 - 0.9$ (technical citation from "l'ingénieur"), the transport length is on the order of millimeters. Beyond this, multiple scattering dominates.

It then becomes impossible with standard optical imaging modalities to construct an optical image at depth. As a reminder, in acousto-optics, the ambition is to image at depths on the order of a few centimeters despite multiple scattering.

The solution to the diffusion equation for a point source in an infinite, homogeneous, and isotropic medium is given by:

$$
\Phi(\mathbf{r}, t) = \frac{1}{(4 \pi D t)^{3/2}} \exp \left( -\frac{r^2}{4 D t} \right)
$$

This corresponds to a 3D Gaussian distribution with variance $2 D t$.

For the subsequent simulations, we will make the following approximation: our optical image in the $XZ$ plane at a depth $Y$ follows a Gaussian distribution.

The optical image corresponds to the cross-section under the ultrasound probe of the optical properties of the medium. It represents the depth image to be reconstructed. The two black spots of absorption illustrate two tumors (with different properties from healthy cells). The halo centered on the image corresponds to the diffusion spot (assuming a Gaussian beam) under the ultrasound probe.

![lambdaAndFFt](https://github.com/user-attachments/assets/40af3b90-b1ec-4038-83e4-4fec8237534b)


### System matrix $A$

the system matrix $A$ is a fundamental component in tomographic reconstruction, serving as the bridge between the unknown properties of the object and the measured data, enabling the reconstruction of images from indirect measurements.

In our specific case of acousto-optic tomography, the system matrix $A$ represents the squared envelope of the acoustic pressure field in the $XZ$ plane where $y=0$. This plane corresponds to the region directly beneath the ultrasound probe.

Example : 

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/e7075b88-2dce-4a35-9c58-85056eebc957" alt="test" width="400"/>
</div>


### Acousto-Optic Signal (AO Signal)

The optical refractive index $n$ is proportional to the mass density $\rho$ to the first order. Thus, as the ultrasonic wave propagates through the medium, it locally alters its refractive index.

By introducing the adiabatic piezoelectric coefficient:

$$
\mu = \frac{\delta n}{\delta \rho}
$$

This modulation of the refractive index is described by the equation:

$$
n(x, y, z, t) = n_0 + \mu P_m \sin \left( 2 \pi f_{US} t - K_{US} z \right)
$$

Where $n_0$ is the average refractive index of the medium and $\Delta n$ is the amplitude of the refractive index modulation.

Assuming a monochromatic plane light wave, the electric field of the emitted wave is of the form:

$$
E(x, y, z, t) = E_M e^{j \left( 2 \pi f_i t - n K_i z \right)}
$$

With $E_M$ the amplitude of the electric field, $f_i$ the frequency of the light, and $K_i$ the norm of the wave vector in the vacuum given by $K_i = 2 \pi \frac{f_i}{c}$.

When the monochromatic plane light wave passes through the acoustic field (of thickness $e$) modulated by the ultrasonic wave, it undergoes diffraction due to the periodic variation of the refractive index.

$$
L = n(x, y, z, t) \cdot e
$$

Where $n(x, y, z, t)$ is the function governing the modification of the refractive index of the medium:

$$
L = n_0 e + \mu P_m \sin \left( 2 \pi f_{US} t - K_{US} y \right) \cdot e
$$

As a reminder, the phase of the light in a given medium is defined by:

$$
\varphi = \frac{2 \pi}{\lambda_0} L_0 = \frac{2 \pi}{\lambda_0} d
$$

Where $\lambda_0$ is the wavelength of light in a vacuum. And in a vacuum, the optical path $L_0$ is simply the physical distance $d$ traveled by the light, as the refractive index of the vacuum is 1.

When the light wave passes through the acoustic field, the phase of the light wave is directly affected by the variation of the refractive index of the medium and by the variation of the optical path. Indeed,

$$
\delta \varphi = \frac{2 \pi}{\lambda_0} L = \frac{2 \pi}{\lambda_0} n(x, y, z, t) e
$$

Thus,

$$
\varphi(x, y, z, t) = \frac{2 \pi}{\lambda_i} n_0 e + \frac{2 \pi \mu}{\lambda_i} P_m \sin \left( 2 \pi f_{US} t - K_{US} y \right)
$$

Let,

$$
\varphi_0 = \frac{2 \pi n_0 e}{\lambda_i} \text{ and } \delta \varphi = \frac{2 \pi \mu P_m e}{\lambda_i}
$$

The electric field of the light wave exiting the medium traversed by the acoustic field is given by:

$$
E(x, y, z+e, t) = E_M e^{j \left( 2 \pi f_i t - n K_i z \right)} \cdot e^{j \varphi_0} e^{j \delta \varphi \sin \left( 2 \pi f_{US} t - K_{US} y \right)}
$$

Expanding in a second-order Taylor series:

$$
E(x, y+e, z, t) \approx E_M e^{j \varphi_0} \left[ \left( 1 - \frac{\delta \varphi^2}{4} \right) e^{j(2 \pi f_i t - n K_i z)} + \frac{\delta \varphi}{2} \left( e^{j 2 \pi (f_i + f_{US}) t - (n K_i z + K_{US} y)} - e^{j 2 \pi (f_i - f_{US}) t - (n K_i z - K_{US} y)} \right) \right]
$$

Where $\left( 1 - \frac{\delta \varphi^2}{4} \right) e^{j \left( 2 \pi f_i t - n K_i y \right)}$ corresponds to the main spectral component of the unmodulated light exiting the acoustic field at $f_i$ with high amplitude $\left( 1 - \frac{\delta \varphi^2}{4} \right)$, and $\frac{\delta \varphi}{2} \left( e^{j 2 \pi (f_i + f_{US}) t - (n K_i z + K_{US} y)} - e^{j 2 \pi (f_i - f_{US}) t - (n K_i z - K_{US} y)} \right)$ represents the two spectral components of the light modulated by the acoustic field at $\pm f_{US}$ with low amplitude $\left( \pm \frac{\delta \varphi}{2} \right)$ (which depends on $P_m$).

The $n$-th order Taylor series expansion would therefore involve the harmonics of the acoustic field $n f_{US}$, where the amplitudes of the harmonics depend on the thickness of the diffraction regime: thin (Raman-Nath) or thick (Bragg). This is developed in the works of **Citation JM**.

As you will have understood, it is the two components modulated by the acoustic field that interest us in acousto-optic imaging, since these two components contain the information on the position of the acoustic wavefront.

Now, it is necessary to make a link between the intensity of the photons marked by the acoustic field passing through the medium and the local intensity of the light in the medium. It is this local intensity that gives us information on the presence or absence of an early tumor.

$$
I_{tagged} \propto \|E_{tagged}\|^{2}
$$

$$
I_{tagged} \propto \left| \delta \varphi \cdot E_0 e^{j \varphi_0} \cdot e^{j 2 \pi (f_i t \pm f_{US}) t - (n K_i y + K_{US} Z)} \right|^2
$$

$$
I_{tagged} \propto \left| E_{0} \right|^2 \cdot \left| \delta \varphi \right|^2
$$

With $\delta \varphi \propto P \) and \( P = P_m \sin(2 \pi f_{US} t - K_{US} z)$.

Thus, if we integrate locally over the entire $XZ$ plane.

Once we have determined the proportionality link between the intensity of the tagged photons measured by the photodiode (acousto-optic signal) and the local optical intensity in the $XZ$ plane, the question arises of how to filter the main component $f_i$. The latter being of high intensity and its bandwidth covers the components at $f_i \pm f_{US}$.

## Detection by Interferometry of Ultrasounds

**TO BE COMPLETED**

### Digital Holography

**TO BE COMPLETED**

### Photorefractive Holography

Photorefractive holography involves using a photorefractive crystal to capture the interference pattern.

### Plane Waves

The concept of tomography fully comes into its own with the notion of a plane wave.

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/c2aac671-6919-46ff-af7d-2952015b4408" alt="test" width="400"/>
</div>

By applying a delay law to the piezoelectric elements of our ultrasonic probe, the greater the delays between each neighboring piezoelectric element, the more significant the emission angle of the plane wave.

### Structured Waves

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/cc615d02-a886-43e0-8efe-ec7974d8573d" alt="test" width="400"/>
</div>

### Mix Waves

Et : 

$y_{\theta,t} = \sum_{x,z} {[A_{t, z , x, \theta}]}^T \lambda_{x,z}$

# Tomographic reconstruction

Tomography is a method of image reconstruction based on a set of projections acquired at different emission angles $\theta$.

In our case, tomography allows us to reconstruct the light intensity map of the tagged photons in the scattering medium solely from the projections, that is, the acousto-optic signals ($1D$) acquired by the photodiode.

As a reminder, the experimental setup is static; no elements move to acquire different projections. Indeed, the advantage of the method is to send different acoustic waves to obtain different projections.

## Analytic Reconstruction (FBP)

Analytic tomographic reconstruction is the simplest type of reconstruction, based on analytic models of integrals.


### iRadon Inversion Method

The iRadon method is derived using the polar coordinates of the object in the Fourier domain. This method constitutes a generalization of the filtered backprojection (FBP).

Generalization of the Fourier Slice Theorem:

For each angle $\theta$, the acquisition is done by spatially structuring the acoustic field emitted by the US probe. The Fourier decomposition of function $h_0(x')$ is the sum of a positive constant equal to its average over one period $\lambda_s$ and harmonic terms of frequency $f_s$.

$$
h_0(x') \approx \frac{1}{2} + \frac{2}{\pi} \cos(2 \pi f_s x' + \phi)
$$

By performing four distinct measurements with $\phi = 0, \frac{\pi}{2}, \pi, \frac{3\pi}{2}$, we retrieve the Fourier component of the object along direction $x'$ using the linear combination:

$$
s(c_s t, \theta, f_s) = \frac{(s_0 - s_\pi) - i(s_{\frac{\pi}{2}} - s_{\frac{3\pi}{2}})}{2 / \pi}
$$

Taking the Fourier transform of the above equation with respect to variable $c_s t$, we find the following generalized Fourier Slice Theorem (FST) relation:

$$
\mathcal{F}_ {c_s t}(s(c_s t, \theta, f_s)) = \mathcal{F}_{x, z}(I)(f_s \sin \theta + f_s \cos \theta, f_s \cos \theta - f_s \sin \theta) 
$$

The integral is then split into two terms \( \int_{\theta=0}^{\pi} \) and \( \int_{\theta=\pi}^{2\pi} \) followed by a change of variable \( \theta \leftarrow \theta - \pi \) on the second term. Making use of the relation \( \tilde{s}(f_s, \theta + \pi, f_s) = \tilde{s}(f_s, \theta, f_s)^* \), and limiting the integration domain to \( [-\theta_m, \theta_m] \) as it is performed in FBP, the generalized FBP expression is:

$$
I_{\text{rec}}(x, z) = \int_{-\theta_m}^{\theta_m} 2 \Re\left[\int_{\mathbb{R}^+} \tilde{s}(f_s, \theta, f_s) e^{2 i \pi x' f_s} e^{2 i \pi x' f_s} f_s \mathrm{d} f_s\right] \mathrm{d} \theta + \int_{-\theta_m}^{\theta_m} \int_{-f_s}^{f_s} \tilde{s}(f_s, \theta, 0) e^{2 i \pi z' f_s} f_s \mathrm{d} f_s \mathrm{d} \theta 
$$

### iFourier Inversion Method

The iFourier method is derived using Cartesian coordinates in the Fourier domain. It is well adapted to the ideal case where US waves are structured for multiple discrete values of \( f_s \) while \( \theta \) is fixed.

Inverse Fourier Transform:

We start with the integral, which defines the inverse Fourier transform using the generalized FST relation, and define the rotation operator of angle $\( \theta \)$:

$$
\mathbf{R}_{\theta}:(f_s, f_s) \in \mathbb{R}^2 \rightarrow \begin{cases}
f_x(f_s, f_s) = f_s \sin \theta + f_s \cos \theta \\
f_z(f_s, f_s) = f_s \cos \theta - f_s \sin \theta
\end{cases}
$$

$\( \mathbf{R}_{\theta} \)$ is a unitary operator, and therefore the determinant of its Jacobian matrix is equal to one. This allows a simple change of integration variables in the following expression:

$$
\mathcal{F}_{f_s, f_s}^{-1}[s(f_s, \theta, f_s)] = \int_{\mathbb{R}^2} \tilde{s}(f_s, \theta, f_s) e^{2 i \pi(x f_x + z f_z)} \mathrm{d} f_x \mathrm{d} f_z = I(x', z')
$$

### Reconstruction Formula

The inversion formula is therefore:

\[ I_{\text{rec}}(x, z) = \frac{1}{N_{\theta}} \sum_{-\theta_m}^{\theta_m} \left[\int_{\mathbb{R}^2} \tilde{s}(f_s, \theta, f_s) e^{2 i \pi(x' f_x + z' f_z)} \mathrm{d} f_x \mathrm{d} f_z\right] \]



## Algebraic reconstruction (MLEM)

## Bayesian reconstruction (MAP)


$$
\theta_j^{(p+1)} = \theta_j^{(p)} + 
\textcolor{blue}{
\frac{
    \theta_j^{(p)} 
}{
    \sum_{i=1}^I a_{ij} 
    \textcolor{green}{+} \textcolor{green}{\theta_j^{(p)} \beta \left. \frac{\partial^2 U}{\partial \theta_j^2} \right|_ {\theta_j = \theta_j^{(p)}}}
}
}
\left(
    {\sum_{i=1}^{I} a_{ij}} \frac{m_i - (\sum_{l=1}^{J} a_{il} \theta_l^{(p)} + b_i)}{\sum_{l=1}^{J} a_{il} \theta_l^{(p)} + b_i}
    \textcolor{green}{-} \textcolor{green}{\beta \left. \frac{\partial U}{\partial \theta_j} \right|_{\theta_j = \theta_j^{(p)}}}
\right)
$$

$$
\hat{\boldsymbol{\theta}}_ {\text{MAP}} = \mathop{\arg\max}\limits_{\boldsymbol{\theta} \in \boldsymbol{\Theta}}
\left( l(\mathbf{m} \mid \boldsymbol{\theta}) - \beta \sum_{j=1}^J \sum_{k \in N_j, k > j} \omega_{kj} \psi(\theta_k, \theta_j) \right)
$$


$$
U(\boldsymbol{\theta}) = \sum_{j=1}^J \sum_{k \in N_j, k > j} \omega_{kj} \boxed{\psi(\theta_k, \theta_j)} 
$$


- Quadratic

$$
\psi_{\text{Quad}} (\theta_k, \theta_j) = \frac{1}{2} \left( \frac{\theta_k - \theta_j}{\sigma} \right)^2
$$

- Huber

$$
  \psi_{\text{Huber}} (\theta_k, \theta_j) =
  \begin{cases}
  \delta |\theta_k - \theta_j| - \frac{\delta^2}{2} & \text{if } |\theta_k - \theta_j| > \delta, \\
  \frac{1}{2} (\theta_k - \theta_j)^2 & \text{if } |\theta_k - \theta_j| \leq \delta.
  \end{cases}
$$

- Relative difference 

$$
    \psi_{\text{RD}} (\theta_k, \theta_j) = 
    \frac{(\theta_k - \theta_j)^2}{(\theta_k + \theta_j) + \gamma |\theta_k - \theta_j|}
$$

$$
\hat{\boldsymbol{\theta}}_  {\text{MAP}} = \mathop{\arg\max}\limits_{\boldsymbol{\theta} \in \boldsymbol{\Theta}} \left( l(\mathbf{m} \mid \boldsymbol{\theta}) - \beta \sum_{j=1}^J \sum_{k \in N_j, k > j} \omega_{kj} \boxed{\psi(\theta_k, \theta_j)} \right)
$$

$$
\hat{\boldsymbol{\theta}}_ {\text{MAP}}  = \mathop{\arg\max}\limits_{\boldsymbol{\theta} \in \boldsymbol{\Theta}}
\left( \sum_{i=1}^I \left( m_i \ln \left( \sum_{j=1}^J a_{ij} \theta_j + b_i \right) - \left( \sum_{j=1}^J a_{ij} \theta_j + b_i \right) - \ln (m_i!) \right) - \beta \sum_{j=1}^J \sum_{k \in N_j, k > j} \omega_{kj} \boxed{\psi(\theta_k, \theta_j)} \right) \\
$$

$$
\hat{\boldsymbol{\theta}}_ {\text{MAP}}  = \mathop{\arg\max}\limits_{\boldsymbol{\theta} \in \boldsymbol{\Theta}}
\left( \sum_{i=1}^I \left( m_i \ln \left( \sum_{j=1}^J a_{ij} \theta_j + b_i \right) - \left( \sum_{j=1}^J a_{ij} \theta_j + b_i \right) \right) - \beta \sum_{j=1}^J \sum_{k \in N_j, k > j} \omega_{kj} \boxed{\psi(\theta_k, \theta_j)} \right)
$$



![influenceIteration](https://github.com/user-attachments/assets/c01c658d-f438-4b7b-8df7-166e1ff7bc13)
![influenceIteration2](https://github.com/user-attachments/assets/1540cf73-bd87-4546-b7d9-e2e6a6b4b0c2)
