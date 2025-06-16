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
systemPath = "/path/to/folder/system_matrixParams.txt"

param = AOT_biomaps.Settings.Params(paramPath)
```





