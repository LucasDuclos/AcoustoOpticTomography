# <u><strong><span style="color:red">Acoustic Simulation Tools Validation</span></strong></u>

Field II and k-Wave are two distinct acoustic simulation tools. This document aims to select an acoustic simulation library by comparing them to experimental data acquired using a hydrophone.

## Propagation principle

### FieldII

Field II uses the concept of spatial impulse responses to simulate the propagation of ultrasonic waves, an approach developed by Tupholme and Stepanishen. This method relies on linear systems theory to determine the ultrasonic field, whether for pulsed or continuous waves. The spatial impulse response gives the ultrasonic field emitted at a specific point in space as a function of time when the transducer is excited by a Dirac function. The field for any type of excitation can then be found by convolving the spatial impulse response with the excitation function.

Field II is primarily used for the simulation of ultrasonic transducers and is widely recognized for its precision in modeling acoustic fields in homogeneous media. However, it is less suited for simulations in heterogeneous or nonlinear media.

### k-wave

k-Wave uses partial differential equations to describe the propagation of acoustic waves. These equations are solved numerically using finite difference methods or pseudo-spectral methods, which allow the simulation of wave propagation in heterogeneous and nonlinear media. k-Wave divides the simulation domain into a grid of points and calculates the temporal evolution of acoustic pressure at each grid point.

k-Wave is particularly suited for simulations in complex and nonlinear media. It is also optimized for GPU use, allowing for faster and more efficient simulations. Additionally, k-Wave benefits from an active community and is constantly evolving, making it more current and better supported.

## Simulation programming

The structure of the two libraries is broadly similar. It is necessary to define the parameters of the probe (number of elements, widths of the elements, heights of the elements) and the simulation (speed of sound, acquisition frequency, grid, etc.). An important point is that both libraries easily integrate the structuring of the wave linked to the activation of the piezoelectric elements (for Field II, the trick of a null apodization window is used to turn off the element) and the delay in the emissions of the elements.

## Experimental method

Two different acoustic fields (a plane wave at 0° and a structured wave at -10°) are emitted from the Aixplorer system into a water tank. A hydrophone with an 85 µm aperture records the local pressure as a function of time for each point on the grid. The hydrophone moves using an automatic mechanical system with a step of 0.3 mm in a 3D XYZ volume (40 mm, 4 mm, 37 mm).

This experimental method allows for the collection of precise data on the propagation of acoustic waves, which can then be compared to the results of simulations performed with Field II and k-Wave. This comparison will determine which acoustic simulation library offers the best results in terms of precision and fidelity to the experimental data.

![Capture d’écran 2025-05-22 150209](https://github.com/user-attachments/assets/8df78298-0204-4ec2-b328-22744060176c)

## Post-Processing adjustment of the volume acquired by the hydrophone

The positions of the probe and the hydrophone do not allow for 2D acquisition in the XZ plane of the acoustic field. Indeed, a slight misalignment is enough for the acquisition to be out of plane. For the sake of time, we performed the 3D acquisition in a volume with a spatial resolution of 0.3 mm in each direction (instead of 0.2 mm for the simulations).

For each axis of the 3D volume, the angle is corrected using maximum pressure maps.

### Axis X: 

The angle is calculated by first determining the coordinate differences ΔY and ΔZ between the barycenter and the vertical axis. Then, the inverse tangent function is used to obtain the angle θ = arctan(ΔZ/ΔY).
![angleX](https://github.com/user-attachments/assets/28196d81-b9ee-46c6-9e6b-f0e7c0afb016)


### Axis Y:

The horizontal position differences, d1 and d2, are calculated by subtracting the start and end positions. The vertical position difference, dz, is obtained by subtracting the heights of the horizontal lines. The angle is then calculated using the inverse tangent function on the ratio of the horizontal and vertical position differences, and then converted to degrees.
![angleY_1](https://github.com/user-attachments/assets/8aa00f55-9951-4168-8b83-ba803f726fff)

### Axis Z: 

The angle is calculated using linear regression on the coordinates of the non-zero pixels in the masked pressure sub-matrix. The sub-matrix is obtained by applying a threshold to reveal the ends of the probe in the XY plane. The slope of the regression line is used to determine the misalignment angle relative to the vertical.
![angleZ_1](https://github.com/user-attachments/assets/72ad14f3-21be-4230-a58f-d8fa3561188a)

## Results

### 2D XZ Slice for $Y=0$


Similarly for the structured wave:


Field II has a significant advantage in this context. It integrates the curvature of the probe in the elevation plane into its reasoning, a feature that is not yet available in k-Wave, where this curvature is set to infinity. In our specific case, since we integrate the calculation of the acousto-optic signal along the Y-axis, the curvature of the probe in the elevation plane (Y) is not important. Therefore, for a 2D slice at a given Y, Field II is the most suitable and offers more precise results.


### 2D XZ Slice for $\sum Y$

This slice is less intuitive but corresponds more closely to our acousto-optic model. In our case, the marked photons are the result of the integration along the Y-axis of the product between the photon flux and the square of the acoustic field envelope. By summing along the Y-axis of the acoustic field, we constrain our 3D problem to a 2D plane.

Additionally, by summing along the Y-axis, we eliminate problems related to the inclination of the probe, as studied in the previous section. This approach simplifies the analysis and allows us to focus on the essential aspects of the acousto-optic model without being affected by variations in the elevation plane.

The attenuation effect of the acoustic field in water is easily visualized in both cases.

## Conclusion 

Field II requires a professional version for faster simulations but does not integrate GPU computing, unlike k-Wave, which constitutes a significant advantage for the latter.

Field II adopts a simpler approach than k-Wave, giving it the advantage of being less computationally expensive (though limited to CPU). For simulations in homogeneous and linear 2D media with specific probes, Field II is an ideal solution. However, if one wishes to integrate heterogeneity into the medium or simulate in a 3D plane, k-Wave is more suitable.

Overall, k-Wave offers better results for simulations. Developed in Python and optimized for GPU, this library is constantly evolving, making it more current. Additionally, it benefits from a more active community. For these reasons, we have chosen to focus on k-Wave for the development of our acoustic simulations.
