# TomographyAcoustoOptic
Acosuto-optic tomographic reconstruction 

## Utilisation de la librairie aot-biomaps:
### Installation de la librairie

CPU : 
```
!pip install --upgrade aot-biomaps
```
ou 
```
!pip install --upgrade aot-biomaps[cpu]
```

GPU :
```
!pip install --upgrade aot-biomaps[gpu]
import torch
!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/{torch.__version__}.html

```
Remarque:
Bien s'assurer que cuda est disponible sur la machine.

```
import AOT_biomaps
print(AOT_biomaps.__version__)
print(AOT_biomaps.__process__)
```

La variable `AOT_biomaps.__process__` retourne le type de process CPU ou GPU sur laquelle la machine va effectuer les calculs

### Set-up des paramètres

```
fieldDir = "/path/to/folder/Fieldfolder"
paramPath = "/path/to/folder/parameters.yaml"
systemPath = "/path/to/folder/System_matrixParams.txt"

param = AOT_biomaps.Settings.Params(paramPath)
```

L'objet ```param``` contient les élements: general / acoustic / optic / reconstruction. Pour plus d'info sur la structure et la définition de chaques paramètres, consulter l'exemple ```ExampleParameters.yaml``` et ```ExampleSystem_matrixParams.txt```.
Pour accéder à un paramètre spécifique `param.acoustic['f_US']`.

### AOT_Experiment

La classe `AOT_Experiment` permet de gérer notre expérience. Lors de sa création avec son constructeur, elle prend en compte les 3 éléments clés:
 - L'image optique
 - Les champs acoustiques
 - Les signaux acousto-optiques

Exemple d'utilisation : 

```
manip = AOT_biomaps.AOT_Experiment.Tomography(params=param, fieldDataPath=fieldDir, fieldParamPath=systemPath)
```

Remarque:

La simulation des champs acoustiques peut faire apparaitre des artefacts au niveau des bords de la grille de simulation. Il peut être nécessaire de tronquer les champs acoustiques : 

```
manip.cutAcousticFields(min_t=0,max_t=float(2.5e-5),saveFields=True)
```
`t_min` et `t_max`sont initialiser en secondes.
Si `saveFields=True`, les champs tronqués sont sauvegardés dans le répertoire.

### AOT_Optic

```
phantom = AOT_biomaps.AOT_Optic.Phantom(params=param)
```







