# AOT-BioMaps

Tomographic reconstruction for acousto-optic imaging

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

La classe `AOT_Experiment` permet de gérer notre expérience. Elle intègre les trois éléments clés de l'imagerie acousto-optique :
 - L'image optique
 - Les champs acoustiques
 - Les signaux acousto-optiques

La librairie prend en compte les deux modes d'utilisations possibles:
 - Simulation
 - Expérimental

La simulation nécessite de générer une image optique (pour la simulation de signaux acousto-optique) et les signaux acousto-optiques. De son côté la tomographie avec des données expérimental nécessite uniquement le chargement des signaux acousto-optiques.

#### Simulation:

```
manip = AOT_biomaps.AOT_Experiment.Tomography(params=param)
manip.generatePhantom()
manip.generateAcousticFields(fieldDataPath, systemPath, show_log = False)
manip.generateAOsignal(withTumor=True)
```

#### Expérimental

```
manip = AOT_biomaps.AOT_Experiment.Tomography(params=param)
manip.generateAcousticFields(fieldDataPath, systemPath, show_log = False)
manip.loadAOsignal(withTumor=True)
```

Remarque:

La simulation des champs acoustiques peut faire apparaitre des artefacts au niveau des bords de la grille de simulation. Il peut être nécessaire de tronquer les champs acoustiques : 

```
manip.cutAcousticFields(min_t=0,max_t=2.5e-5,saveFields=True)
```
`t_min` et `t_max`sont initialisés en secondes.
Si `saveFields=True`, les champs tronqués sont sauvegardés dans le répertoire.

### Reconstruction

- Analytique
- Algébrique
- Bayésienne

#### Analytique

#### Algébrique

Il existe différent algorithme 
Par défaut, la reconstruction s'effectue avec un optimiseur Maximum Likelihood Estimation Method (ML-EM) (pour plus d'information regarder la documentation)

```
optimizer =  AOT_biomaps.AOT_Reconstruction.OptimizerType.MLEM

recon = AOT_biomaps.AOT_Reconstruction.AlgebraicRecon(experiment= manip, opti=optimizer, numIterations=200,saveDir=f"/home/duclos/AOT/SetMixte/{set}/recon",isGPU=False)
recon.run()
```

#### Bayésienne

Pour l'instant uniquement les optimiseurs suivants sont supportés par la librairie:
 - Preconditioned Conjugate Gradient Maximum A Posteriori Expectation Maximization (**PCG MAP-EM**)
 - Preconditioned Conjugate Gradient Maximum A Posteriori Expectation Maximization avec condition stop (**PCG MAP-EM stop**)
 - De Pierro Maximum A Posteriori Expectation Maximization (**Pierro MAP-EM**)

Pour l'instant uniquement les fonctions potentielles suivantes sont supportées par la librairie:
 - Huber (`AOT_biomaps.AOT_Reconstruction.PotentialType.HUBER_PIECEWISE`)
 - Quadratique (`AOT_biomaps.AOT_Reconstruction.PotentialType.QUADRATIC`)
 - Différence Relative (`AOT_biomaps.AOT_Reconstruction.PotentialType.RELATIVE_DIFFERENCE`)


```
optimizer =  AOT_biomaps.AOT_Reconstruction.OptimizerType.PGC
potentialFunction = AOT_biomaps.AOT_Reconstruction.PotentialType.HUBER_PIECEWISE

recon = AOT_biomaps.AOT_Reconstruction.BayesianRecon(experiment=manip, opti=optimizer, potentialFunction=potentialFunction, numIterations=200,saveDir=f"/home/duclos/AOT/SetMixte/{set}/recon",isGPU=False)
recon.run()
```






