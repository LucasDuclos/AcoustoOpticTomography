# AOT_Acoustic

The `AOT_biomaps.AOT_Acoustic` module groups all classes associated with the creation of acoustic fields, intended for acousto-optic tomography. The various available ultrasonic emissions are as follows:

- Focused Waves (`AOT_biomaps.AOT_Acoustic.FocusedWave()`)
- Plane Waves (`AOT_biomaps.AOT_Acoustic.PlaneWave()`)
- Structured Waves (`AOT_biomaps.AOT_Acoustic.StructuredWave()`)
- Irregular Waves (`AOT_biomaps.AOT_Acoustic.IrregularWave()`)

It is possible to simulate acoustic fields in 2D (`AOT_biomaps.AOT_Acoustic.Dim.D2`) and 3D (`AOT_biomaps.AOT_Acoustic.Dim.D3`). As explained in the theory, the acquisition of fields in 3D is not necessary (kwave does not take into account the elevation radius of the probe). Note that, like other modules in the library, the creation of acoustic fields can be done with either CPU or GPU.

## General Functions

### Generate Field (2D and 3D)

### Calculate Squared Envelope of the Acoustic Field

This function calculates the squared analytical envelope of an acoustic field using either the CPU or the GPU.

#### Parameters:
- `isGPU`: Boolean indicating whether the calculation should be performed on the GPU (default is `True` if the configured process is 'gpu', otherwise `False`).

#### Returns:
- `envelope`: A `numpy.ndarray` or `cupy.ndarray` representing the squared analytical envelope of the acoustic field.

**Hilbert Transformation:**

- For each slice \( A_i \) of the acoustic field, calculate the analytical envelope \( \mathcal{H}(A_i) \) using the Hilbert transform:
 
$$
  \mathcal{H}(A_i)(t) = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{A_i(\tau)}{t - \tau} \, d\tau
$$

- Calculate the squared envelope:

$$
  |\mathcal{H}(A_i)(t)|^2
$$

### Save Field

There are three save formats available in the AOT-Biomaps library. The three formats allow saving the pressure values of the created acoustic field.

- HDR/IMG Format (`AOT_biomaps.AOT_Acoustic.FormatSave.HDR_IMG`)

  Saving in HDR/IMG format is the default save format in the library. When saving, two files (`.hdr` and `.img`) are created. The HDR file contains the characteristics of the acoustic fields in ASCII format, and the IMG file contains the acoustic field in `float32` format.

- HDF5 Format (`AOT_biomaps.AOT_Acoustic.FormatSave.H5`)

- Numpy Array Format (`AOT_biomaps.AOT_Acoustic.FormatSave.NPY`)

## Focused Waves

## Plane Waves

$$
\text{delay}[i] = \frac{x_i \cdot \tan(\theta)}{c_0}
$$

## Structured Waves

## Irregular Waves
