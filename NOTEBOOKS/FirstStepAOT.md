Premiers pas en tomographie acousto-optique

![schéma_tomo_AVEC_particules2](https://github.com/user-attachments/assets/b8ebaed4-4c7a-4dea-80f0-039e9902b7ea)

### Optic Image $\lambda$

The optical image corresponds to the cross-section under the ultrasound probe of the optical properties of the medium. It represents the depth image to be reconstructed. The two black spots of absorption illustrate two tumors (with different properties from healthy cells). The halo centered on the image corresponds to the diffusion spot (assuming a Gaussian beam) under the ultrasound probe.

![LAMBDA](https://github.com/user-attachments/assets/301aa68c-3c9c-46ba-83c1-ba8a87c7b69f)

### System matrix $A$

the system matrix $A$ is a fundamental component in tomographic reconstruction, serving as the bridge between the unknown properties of the object and the measured data, enabling the reconstruction of images from indirect measurements.

In our specific case of acousto-optic tomography, the system matrix $A$ represents the squared envelope of the acoustic pressure field in the $XZ$ plane where $y=0$. This plane corresponds to the region directly beneath the ultrasound probe.

Example : 

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/e7075b88-2dce-4a35-9c58-85056eebc957" alt="test" width="400"/>
</div>


### Acousto-optic signal (AO Signal)

$$
    y_{\theta,t} = \sum_{x,z} {[A_{t, z , x, \theta}]}^T \lambda_{x,z}
$$


