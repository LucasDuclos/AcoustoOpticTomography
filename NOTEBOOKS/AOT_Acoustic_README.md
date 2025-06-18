# AOT_Acoustic

Le module `AOT_biomaps.AOT_Acoustic` regroupe toutes les classes associées à la création de champs acoustiques, destiné à la tomographie acousto-optique. Les différentes émissions ultrasonores disponibles sont les suivantes:

- Focused Waves (`AOT_biomaps.AOT_Acoustic.FocusedWave()`)
- Plane Waves (`AOT_biomaps.AOT_Acoustic.PlaneWave()`)
- Structured Waves (`AOT_biomaps.AOT_Acoustic.StructuredWave()`)
- Irregular Waves (`AOT_biomaps.AOT_Acoustic.IrregularWave()`)

Il possible de simuler des champs acoustique en 2D (`AOT_biomaps.AOT_Acoustic.Dim.D2`) 3D (`AOT_biomaps.AOT_Acoustic.Dim.D3`). Comme expliqué dans la théorie, l'acquisition de champs en 3D n'est pas nécessaire (kwave ne prend pas en compte le rayon d'élévation de la sonde). À noter que comme les autres module de la librairie, la création de champs acoustiques peut se faire avec CPU ou GPU.

## General functions

### Generate Field (2D and 3D) 

### Calculate squared enveloppe of the acoustic Field

Cette fonction calcule l'enveloppe analytique au carré d'un champ acoustique en utilisant soit le CPU soit le GPU.

#### Paramètres :

- `isGPU` : Booléen indiquant si le calcul doit être effectué sur le GPU (par défaut, `True` si le processus configuré est 'gpu', sinon `False`).
#### Retourne :
- `envelope` : Un tableau `numpy.ndarray` ou `cupy.ndarray` représentant l'enveloppe analytique au carré du champ acoustique.

**Transformation de Hilbert :**
   - Pour chaque tranche \( A_i \) du champ acoustique, calcule l'enveloppe analytique \( \mathcal{H}(A_i) \) en utilisant la transformation de Hilbert :

$$
\mathcal{H}(A_i)(t) = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{A_i(\tau)}{t - \tau} \, d\tau
$$
     
   - Calcule l'enveloppe au carré :

$$
\mathcal{H}(A_i)(t)|^2
$$
     
  
### Save Field

Il existe trois formats de sauvegarde disponibles dans la librairie AOT-Biomaps. Les trois formats permettent la sauvegarde des valeurs de pression du champ acoustique créé.

- Format HDR / IMG (`AOT_biomaps.AOT_Acoustic.FormatSave.HDR_IMG`)

La sauvegarde sous le format HDR / IMG est la sauvegarde par défaut dans la librairie.
Lors de la sauvegarde, deux fichiers (`.hdr`) et (`.img`) sont créés. Le fichier HDR contient au format ASCII les caractéristiques du champs acoustiques, le fichier IMG contient le champ acoustique en format `float32`.
- Format HDF5 (`AOT_biomaps.AOT_Acoustic.FormatSave.H5`)

- Format Numpy Array (`AOT_biomaps.AOT_Acoustic.FormatSave.NPY`)

## Focused Waves

## Plane Waves

$$
\text{delay}[i] = \frac{x_i \cdot \tan( \theta|)}{c_0}
$$

## Structured Waves

## Irregular Waves


