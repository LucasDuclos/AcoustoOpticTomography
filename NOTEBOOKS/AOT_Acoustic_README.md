# AOT_Acoustic

Le module `AOT_Acoustic` regroupe toutes les classes associées à la création de champs acoustiques, destiné à la tomographie acousto-optique. Les différentes émissions ultrasonores disponibles sont les suivantes:

- Focused Waves (`AOT_biomaps.AOT_Acoustic.FocusedWave()`)
- Plane Waves (`AOT_biomaps.AOT_Acoustic.PlaneWave()`)
- Structured Waves (`AOT_biomaps.AOT_Acoustic.StructuredWave()`)
- Irregular Waves (`AOT_biomaps.AOT_Acoustic.IrregularWave()`)

Il possible de simuler des champs acoustique en 2D (`AOT_biomaps.AOT_Acoustic.Dim.D2`) 3D (`AOT_biomaps.AOT_Acoustic.Dim.D3`). Comme expliqué dans la théorie, l'acquisition de champs en 3D n'est pas nécessaire (kwave ne prend pas en compte le rayon d'élévation de la sonde). 

## Plane Waves (`PlaneWave()`)

$$
\text{delay}[i] = \frac{x_i \cdot \tan( \theta|)}{c_0}
$$
