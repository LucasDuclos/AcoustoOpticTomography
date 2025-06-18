# AOT_Acoustic

Le module `AOT_biomaps.AOT_Acoustic` regroupe toutes les classes associées à la création de champs acoustiques, destiné à la tomographie acousto-optique. Les différentes émissions ultrasonores disponibles sont les suivantes:

- Focused Waves (`AOT_biomaps.AOT_Acoustic.FocusedWave()`)
- Plane Waves (`AOT_biomaps.AOT_Acoustic.PlaneWave()`)
- Structured Waves (`AOT_biomaps.AOT_Acoustic.StructuredWave()`)
- Irregular Waves (`AOT_biomaps.AOT_Acoustic.IrregularWave()`)

Il possible de simuler des champs acoustique en 2D (`AOT_biomaps.AOT_Acoustic.Dim.D2`) 3D (`AOT_biomaps.AOT_Acoustic.Dim.D3`). Comme expliqué dans la théorie, l'acquisition de champs en 3D n'est pas nécessaire (kwave ne prend pas en compte le rayon d'élévation de la sonde). 

## General functions

### Generate Field (2D and 3D) 

### Calculate squared enveloppe of the acoustic Field

### Save Field

Il existe trois formats de sauvegarde disponibles dans la librairie AOT-Biomaps.

- Format HDR / IMG (`AOT_biomaps.AOT_Acoustic.FormatSave.HDR_IMG`)

La sauvegarde sous le format HDR / IMG est la 

- Format HDF5 (`AOT_biomaps.AOT_Acoustic.FormatSave.H5`)

- Format Numpy Array (`AOT_biomaps.AOT_Acoustic.FormatSave.NPY`)

## Focused Waves

## Plane Waves

$$
\text{delay}[i] = \frac{x_i \cdot \tan( \theta|)}{c_0}
$$

## Structured Waves

## Irregular Waves


