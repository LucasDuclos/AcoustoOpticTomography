<h1 style="text-align:center; color:red; font-weight:bold; text-decoration:underline;">
    VALIDATION OF SIMULATED ACOUSTIC FIELDS
</h1>

This notebook aims to validate our methods for simulating acoustic fields (K-Wave/Field II) in the case of inclined plane and structured waves. To validate these fields, we compared the simulations with acoustic fields acquired by a hydrophone in a water tank.

Given the time required to acquire acoustic data at each point in the 3D volume of our acoustic field with the hydrophone, we validate the simulations based on two specific configurations:

- **Plane wave at $(0^\circ)$**
- **Structured wave at $(-10^\circ)$**

The data is acquired across the entire 3D volume with the following parameters:
- $(x \in [-21, 21])$ with a step of 0.3 mm
- $(y \in [-2, 2])$ with a step of 0.3 mm
- $(z \in [0, 37])$ with a step of 0.3 mm

To address the uncertainty regarding the probe's position, we acquire the acoustic field in the 3D plane and then perform a summation along the \(Y\) axis. This allows us to obtain only the envelope of the acoustic field in the \(XZ\) plane.

The data is qualitatively validated using the following elements:
- **Spatial maps of the spatial maxima of the squared acoustic field envelope**, calculated over the spatial dimensions \(X\) and \(Z\) for each time instant. This is represented by:

  $$
  M = \underset{t}{\max} \left| A(t, X, Z) \right|^2
  $$

- **Movie of the squared acoustic envelopes**, represented by:

  $$
  F(t) = \left| A(t, X, Z) \right|^2
  $$

# I. Analysis of Ultrasound Propagation Speed in the Tank with the Hydrophone

## Objective:

To closely match the conditions in the hydrophone tank, it is necessary to calculate the propagation speed of ultrasonic waves in the medium.

## Method

1. **Loading Data**:
   - Load acoustic data from `.h5` and `.mat` files.

2. **Preprocessing**:
   - Sum the intensities along the Z-axis to obtain a 2D representation.

3. **Wavefront Detection**:
   - Identify the positions of the wavefronts at times $t_1$ and $t_2$ by finding the indices of the intensity maxima.

4. **Calculating Distance and Time**:
   - Convert positions $z_1$ and $z_2$ from pixels to meters.
   - Convert time indices $t_1$ and $t_2$ to seconds.

## Speed Calculation

- **Formula**:
  $$
  \text{Speed} = \frac{\text{Distance traveled}}{\text{Time elapsed}}
  $$

- **Distance traveled**:
  $$
  \Delta d = (z_2 - z_1) \times 0.3 \, \text{mm} \times \frac{1}{1000} \, \text{m/mm}
  $$

- **Time elapsed**:
  $$
  \Delta t = (t_2 - t_1) \times 0.04 \, \mu\text{s} \times \frac{1}{1,000,000} \, \text{s/}\mu\text{s}
  $$

- **Speed**:
  $$
  v = \frac{\Delta d}{\Delta t}
  $$

- **Result**:
  - Round the speed to the nearest unit in meters per second (m/s).

 # II. Adjustment of the Probe Position

Since the probe is not perfectly aligned in the plane of the simulated acoustic fields, the fields must be aligned to be compared.

To do this, the data is visualized at the initial time, i.e., time zero, and the position of the wavefront in the measured data is detected at this time. The time delay required for the wave to travel a certain distance is calculated considering the speed of sound, and this delay is converted into the number of samples to synchronize the measured data with the simulations.

# III. Plane Wave Results at $0^{\circ}$

## a. Movie of the Squared Acoustic Envelopes

## b. Cartes spatiales des maxima spatiaux de l'enveloppe du champ acoustique au carré
![PlaneWaveMAX](https://github.com/user-attachments/assets/b89e1d0a-63f4-418d-b4a9-53c375b31f8f)


![PlaneWaveMAX_2](https://github.com/user-attachments/assets/1f252db2-1ed2-415d-9bb5-92c04ad5265f)

# IV. Résultats onde structurée ($f_s=0.31mm^{-1}$) à $-10^{\circ}$

## a. Film des enveloppes acoustiques élevées au carré

## b. Max pressure

![StructuredWaveMAX](https://github.com/user-attachments/assets/1e9bab48-3158-4488-bc8c-b2217c5129ad)


![StructuredWaveMAX_2](https://github.com/user-attachments/assets/75b3838a-f05b-4eeb-ad38-34e510e6b9a5)

# V. Conclusion

Structuration -> OK
Angle -> OK 
Propagation -> bizarre
