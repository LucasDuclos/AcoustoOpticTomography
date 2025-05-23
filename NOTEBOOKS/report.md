
 <span style="color:red; font-weight:bold; text-decoration:underline;">Acousto-Optic</span>

# Introduction


## Formation d'un signal acousto-optic

Comme son nom l'indique, l'acousto-optic est une bimodalité d'imagerie qui permet de combiner l'acoustique et l'optique dans le but de reconstruire une image optique en profondeur dans un milieu diffusant. Ce qui n'est pas possible avec les modlalités d'imagerie optique standard.

### 

Premiers pas en tomographie acousto-optique


![schéma_tomo_AVEC_particules2](https://github.com/user-attachments/assets/b8ebaed4-4c7a-4dea-80f0-039e9902b7ea)

### Optic Image $\lambda$

Dans un milieu diffusant la lumière incidante dévie de sa trajectoire lorsqu'elle rencontre des particules. Dépassé une certaine longueur de transport, le photon incident perd la mémoire de sa direction initiale. 

$l^{*}=\frac{l_s}{1-g}$ où $l_s$ est la distance moyenne entre eux évènement de diffusion.
$g$ définit la directivité de la lumière. Lorsque $g=1$, la lumière se propage en ligne droite. Dans le cas de tissus biologique $g \approx 0.8-0.9$  **citation technique de l'ingénieur**, la longueur de transport est de l'ordre du mm. Au delà, la diffusion multiple prend le dessus.
Il devient alors impossible avec des modalités d'imagerie optique standard de construire une image optique en profondeur.

Pour rappel, en acousto-optique, l'ambition est d'imager à une profondeur de l'ordre de quelques centimètres. La diffusion multiple 

La solution de l'équation de diffusion pour une source ponctuelle dans un milieu infini, homogène et isotrope, est donnée par :

$$
\Phi(\mathbf{r}, t) = \frac{1}{(4 \pi D t)^{3/2}} \exp \left( -\frac{r^2}{4 D t} \right)
$$

Ce qui correspond à une distribution gaussienne en 3D avec variance $\(2 D t\)$.

Pour la suite des simulations nous allons faire l'approximation suivante, notre image optique dans le plan XZ à une profondeur Y suit une distribution gaussienne.

The optical image corresponds to the cross-section under the ultrasound probe of the optical properties of the medium. It represents the depth image to be reconstructed. The two black spots of absorption illustrate two tumors (with different properties from healthy cells). The halo centered on the image corresponds to the diffusion spot (assuming a Gaussian beam) under the ultrasound probe.

![lambdaAndFFt](https://github.com/user-attachments/assets/40af3b90-b1ec-4038-83e4-4fec8237534b)


### System matrix $A$

the system matrix $A$ is a fundamental component in tomographic reconstruction, serving as the bridge between the unknown properties of the object and the measured data, enabling the reconstruction of images from indirect measurements.

In our specific case of acousto-optic tomography, the system matrix $A$ represents the squared envelope of the acoustic pressure field in the $XZ$ plane where $y=0$. This plane corresponds to the region directly beneath the ultrasound probe.

Example : 

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/e7075b88-2dce-4a35-9c58-85056eebc957" alt="test" width="400"/>
</div>


### Acousto-optic signal (AO Signal)

L'indice de réfraction optique $n$ est proportionnel au premier ordre à la masse volumique de celui-ci $\rho$.

Ainsi, lorsque l'onde ultrasonore se propage dans le milieu, elle modifie localement son indice de réfraction.

En introduisant le coefficient adiabatique piezo-électrique :

$$
\mu = \frac{\delta n}{\delta \rho} 
$$

Cette modulation de l'indice de réfraction est décrite par l'équation : 

$$
n (x,y,z,t)  = n_0 + \mu P_m \sin \left( 2 \pi f_{US} t - K_{US} z \right)
$$

Où $n_0$ est l'indice moyen de réfraction du milieu et $\Delta n$ est l'amplitude de la modulation de l'indice de réfraction.

En supposant une onde plane monochromatique lumineuse, le champ électrique de l'onde émise est de la forme :

$$
E(x,y,z,t) = E_M e^{j \left( 2 \pi f_i t - n K_i z \right)}
$$

Avec $E_M$ l'amplitude du champ électrique, $f_i$ la fréquence de la lumière et $K_i$ la norme du vecteur d'onde dans le vide donné par $K_i = 2 \pi \frac{f_i}{c}$

Lorsque l'onde plane monochromatique lumineuse traverse le champ acoustique (d'épaisseur $e$) modulé par l'onde ultrasonore, elle subit une diffraction due à la variation périodique de l'indice de réfraction. 

$$ 
L = n(x,y,z,t) \dot e 
$$

Où $n(x,y,z,t)$ est la fonction qui régit la modification de l'indice de réfraction du milieu

$$
L = n_0e + \mu P_m \sin \left( 2 \pi f_{US}t-K_{US}y \right) \dot e
$$

Pour rappel, la phase de la lumière dans un milieu donné est définie par: 

$$
\varphi = \frac{2 \pi}{\lambda_0}L_0=\frac{2 \pi}{\lambda_0}d
$$

Où $\lambda_0$ est la longueur d'onde de la lumière dans le vide. Et dans le vide, le chemin optique $L_0$ est simplement la distance physique $d$ parcourue par la lumière, car l'indice de réfraction du vide est 1.

Lorsque l'onde lumineuse traverse le champs acoustique,la phase de l'onde lumineuse est directement affecté par la variation de l'indice de réfraction du milieu et par la variation du chemin optique.

En effet, 

$$
\delta \varphi = \frac{2 \pi}{\lambda_0}L = \frac{2 \pi}{\lambda_0}n(x,y,z,t)e
$$

Ainsi,

$$
\varphi(x,y,z,t)= \frac{2 \pi}{\lambda_i}n_0 e + \frac{2 \pi n}{\lambda_i} P_m \sin \left(2 \pi f_{US} t - K_{US} y \right)
$$

Posons,

$$
\varphi_0 = \frac{2 \pi n_0 e}{\lambda_i} \text{   et,   } \delta \varphi = \frac{2 \pi \mu P_m e}{\lambda_i}
$$

Le champ électrique de l'onde lumineuse en sortie du milieu traversé par le champ acoustique est donné par :

$$
E(x, y, z+e, t) = E_M e^{j\left( 2 \pi f_i t - n K_i z \right)} \cdot e^{j\varphi_0} e^{j\delta \varphi \left( 2 \pi f_{us} t - K_{us} y \right)}
$$

Décomposons en série de taylor du secon ordre:

$$
E(x, y+e, z, t) \approx E_M e^{j \varphi_0} \left[ \left( 1 - \frac{\delta \varphi^2}{4} \right) e^{j(2 \pi f_i t - n K_i z)} + \frac{\delta \varphi}{2} \left( e^{j 2 \pi (f_i + f_{us}) t - (n K_i z + K_{us} y)} - e^{j 2 \pi (f_i - f_{us}) t - (n K_i z - K_{us} y)} \right) \right]
$$



Où $\left( 1-\frac{\delta \varphi^{2}}{4} \right) e^{j\left( 2 \pi f_i t -n K_i y\right)}$ correspond à la composante spectrale principale de la lumière non modulé en sortie du champs acoustique à $f_i$
et $\frac{\delta \varphi}{2} \left( e^{j 2 \pi (f_i + f_{us}) t - (n K_i z + K_{us} y)} - e^{j 2 \pi (f_i - f_{us}) t - (n K_i z - K_{us} y)} \right)$ représente les deux composantes 
spectrale de la lumière modulé par le champs acoustique à $\pm f_{US}$



Et : 

$y_{\theta,t} = \sum_{x,z} {[A_{t, z , x, \theta}]}^T \lambda_{x,z}$

# Tomographic reconstruction

## Analytic reconstruction (FBP)

## Algebraic reconstruction (MLEM)


### Plane Waves

### Structured Waves

### Mix Waves

![influenceIteration](https://github.com/user-attachments/assets/c01c658d-f438-4b7b-8df7-166e1ff7bc13)
![influenceIteration2](https://github.com/user-attachments/assets/1540cf73-bd87-4546-b7d9-e2e6a6b4b0c2)
