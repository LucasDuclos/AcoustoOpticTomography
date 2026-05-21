# Contribution Guide - AOT_biomaps

Merci de votre intérêt pour contribuer à AOT_biomaps ! Ce guide vous explique comment contribuer au projet.

## 📋 Table des matières

- [Code de conduite](#-code-de-conduite)
- [Comment contribuer](#-comment-contribuer)
  - [Signaler un bug](#signaler-un-bug)
  - [Suggérer une fonctionnalité](#suggérer-une-fonctionnalité)
  - [Contribuer au code](#contribuer-au-code)
- [Environnement de développement](#-environnement-de-développement)
- [Style de code](#-style-de-code)
- [Tests](#-tests)
- [Documentation](#-documentation)
- [Pull Requests](#-pull-requests)
- [Reconnaissance](#-reconnaissance)

---

## 🤝 Code de conduite

En participant à ce projet, vous acceptez de respecter le [Code de conduite](CODE_OF_CONDUCT.md).

**Principes principaux** :
- Soyez respectueux et inclusif
- Soyez constructif dans vos critiques
- Respectez les opinions différentes
- Contribuez de manière positive

---

## 🚀 Comment contribuer

### Signaler un bug

Si vous trouvez un bug, veuillez :

1. **Vérifier les issues existantes** : Assurez-vous que le bug n'a pas déjà été signalé.
2. **Créer une issue** avec les informations suivantes :
   - Titre clair et descriptif
   - Description détaillée du problème
   - Étapes pour reproduire le bug
   - Comportement attendu vs. comportement réel
   - Informations sur votre environnement (OS, Python version, dépendances)
   - Exemple de code minimal pour reproduire le bug

**Template pour les bugs** :
```markdown
## Description
[Description claire du bug]

## Étapes pour reproduire
1. [Étape 1]
2. [Étape 2]
3. [Étape 3]

## Comportement attendu
[Ce qui devrait se passer]

## Comportement réel
[Ce qui se passe réellement]

## Environnement
- OS: [Windows/Linux/macOS]
- Python: [version]
- NumPy: [version]
- CuPy: [version ou "non installé"]

## Exemple de code
```python
[Code minimal pour reproduire le bug]
```
```

---

### Suggérer une fonctionnalité

Si vous avez une idée pour améliorer la librairie :

1. **Vérifier les discussions existantes** : Assurez-vous que la fonctionnalité n'a pas déjà été proposée.
2. **Créer une issue** avec :
   - Description de la fonctionnalité
   - Cas d'usage
   - Avantages pour la communauté
   - Exemples de code si applicable

**Template pour les fonctionnalités** :
```markdown
## Description de la fonctionnalité
[Description détaillée]

## Problème résolu
[Quel problème cette fonctionnalité résout-elle ?]

## Cas d'usage
[Dans quelles situations cette fonctionnalité serait-elle utile ?]

## Implémentation proposée
[Idées sur comment implémenter cette fonctionnalité]

## Exemple de code
```python
[Exemple d'utilisation si applicable]
```
```

---

### Contribuer au code

#### 1. Forker le dépôt

1. Forker le dépôt sur GitHub
2. Cloner votre fork localement
   ```bash
   git clone https://github.com/votre-utilisateur/AcoustoOpticTomography.git
   cd AcoustoOpticTomography
   ```

#### 2. Créer une branche

Créez une branche pour votre contribution :
```bash
# Pour une nouvelle fonctionnalité
git checkout -b feature/nouvelle-fonctionnalite

# Pour un correctif
git checkout -b fix/description-du-bug

# Pour de la documentation
git checkout -b docs/amélioration-documentation
```

#### 3. Faire vos modifications

- Faites vos modifications dans la branche
- Respectez le [style de code](#-style-de-code)
- Ajoutez des tests pour vos modifications
- Mettez à jour la documentation si nécessaire

#### 4. Commiter vos changements

```bash
# Ajouter les fichiers modifiés
git add .

# Commiter avec un message clair
git commit -m "Type: Description de la modification"
```

**Convention de commit** :
- `feat:` - Nouvelle fonctionnalité
- `fix:` - Correctif de bug
- `docs:` - Modification de la documentation
- `style:` - Modifications de style (formatage, etc.)
- `refactor:` - Refactorisation du code
- `perf:` - Amélioration des performances
- `test:` - Ajout ou modification de tests
- `chore:` - Autres modifications (configuration, etc.)

#### 5. Pousser vos changements

```bash
# Pousser sur votre fork
git push origin nom-de-votre-branche
```

#### 6. Créer une Pull Request

1. Allez sur le dépôt original sur GitHub
2. Cliquez sur "New Pull Request"
3. Sélectionnez votre branche
4. Remplissez le template de PR
5. Soumettez la PR

---

## 💻 Environnement de développement

### Configuration recommandée

1. **Python** : ≥ 3.8 (recommandé 3.10 ou 3.11)
2. **Environnement virtuel** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows
   ```

3. **Installation en mode développement** :
   ```bash
   pip install -e .
   ```

4. **Installer les dépendances de développement** :
   ```bash
   pip install -r requirements-dev.txt
   ```

### Outils recommandés

- **Éditeur** : VS Code, PyCharm
- **Linter** : flake8, pylint
- **Formatter** : black, isort
- **Tests** : pytest
- **Documentation** : mkdocs, sphinx

---

## 🎨 Style de code

### Conventions générales

1. **Nommage** :
   - Variables et fonctions : `snake_case`
   - Classes : `PascalCase`
   - Constantes : `UPPER_SNAKE_CASE`
   - Méthodes privées : `_prefix`

2. **Documentation** :
   - Toujours documenter les classes et fonctions publiques
   - Utiliser les docstrings au format Google
   - Inclure les types dans les docstrings

3. **Longueur des lignes** : 100 caractères maximum

4. **Encodage** : UTF-8

### Exemple de docstring

```python
"""
Nom de la fonction ou classe.

Description détaillée de ce que fait la fonction/classe.

Args:
    param1 (type): Description du paramètre 1.
    param2 (type, optional): Description du paramètre 2. Defaults to valeur_par_défaut.

Returns:
    type: Description de la valeur de retour.

Raises:
    TypeError: Si le type d'un paramètre est incorrect.
    ValueError: Si une valeur de paramètre est invalide.

Example:
    >>> fonction(1, 2)
    3
"""
```

### Exemple de classe

```python
class MaClasse:
    """Description de la classe."""
    
    def __init__(self, param1: int, param2: str = "default"):
        """
        Initialise une nouvelle instance de MaClasse.
        
        Args:
            param1 (int): Description du paramètre 1.
            param2 (str, optional): Description du paramètre 2. Defaults to "default".
        """
        self.param1 = param1
        self.param2 = param2
    
    def ma_methode(self, param: float) -> float:
        """
        Description de la méthode.
        
        Args:
            param (float): Description du paramètre.
            
        Returns:
            float: Description de la valeur de retour.
        """
        return param * self.param1
```

---

## 🧪 Tests

### Structure des tests

Les tests sont organisés dans le dossier `tests/` :
```
tests/
├── unit/          # Tests unitaires
├── integration/  # Tests d'intégration
└── data/          # Données de test
```

### Écrire des tests

1. **Tests unitaires** : Tester des fonctions individuelles
2. **Tests d'intégration** : Tester l'interaction entre composants
3. **Tests de performance** : Vérifier les performances

### Exemple de test unitaire

```python
import unittest
import numpy as np
from AOT_biomaps.AOT_Recon.AOT_Kernels import fill_array_value

class TestKernels(unittest.TestCase):
    """Tests pour les fonctions utilitaires des noyaux."""
    
    def test_fill_array_value_cpu(self):
        """Test de fill_array_value sur CPU."""
        arr = np.zeros(10)
        fill_array_value(arr, 5.0, device='cpu')
        self.assertTrue(np.all(arr == 5.0))
    
    def test_fill_array_value_gpu(self):
        """Test de fill_array_value sur GPU."""
        try:
            import cupy as cp
            arr = cp.zeros(10)
            fill_array_value(arr, 5.0, device='gpu')
            self.assertTrue(cp.all(arr == 5.0))
        except ImportError:
            self.skipTest("CuPy non disponible")

if __name__ == '__main__':
    unittest.main()
```

### Exécuter les tests

```bash
# Exécuter tous les tests
python -m pytest tests/

# Exécuter un test spécifique
python -m pytest tests/unit/test_kernels.py

# Exécuter avec couverture de code
python -m pytest --cov=AOT_biomaps tests/
```

---

## 📚 Documentation

### Mettre à jour la documentation

La documentation est dans le dossier `docs/` au format Markdown.

1. **Modifiez les fichiers .md** directement
2. **Générez la documentation** (si applicable) :
   ```bash
   mkdocs build
   ```
3. **Vérifiez les liens** : Assurez-vous que tous les liens fonctionnent

### Règles pour la documentation

1. Utilisez un langage clair et simple
2. Incluez des exemples de code
3. Documentez les paramètres et valeurs de retour
4. Ajoutez des diagrammes si utile
5. Mettez à jour le README.md pour les changements majeurs

---

## 🔄 Pull Requests

### Processus de review

1. **Création** : Votre PR est créée
2. **Vérification automatique** : Les tests CI s'exécutent
3. **Review manuelle** : Un mainteneur review votre code
4. **Corrections** : Vous corrigez les problèmes signalés
5. **Fusion** : Votre PR est fusionnée dans la branche main

### Critères d'acceptation

Pour qu'une PR soit acceptée, elle doit :

✅ **Fonctionner** : Le code doit fonctionner comme attendu
✅ **Passer les tests** : Tous les tests existants doivent passer
✅ **Respecter le style** : Le code doit suivre le style du projet
✅ **Être documentée** : Le code doit être documenté
✅ **Avoir des tests** : Les nouvelles fonctionnalités doivent avoir des tests
✅ **Ne pas casser** : Ne pas casser les fonctionnalités existantes

### Template de Pull Request

```markdown
## Description

[Description claire de la modification]

## Type de modification
- [ ] Bug fix
- [ ] Nouvelle fonctionnalité
- [ ] Amélioration de la documentation
- [ ] Refactorisation
- [ ] Amélioration des performances
- [ ] Autres (précisez) : _______

## Modifications apportées
- [ ] Ajout de code
- [ ] Modification de code existant
- [ ] Suppression de code
- [ ] Modification de la documentation
- [ ] Ajout de tests

## Tests
- [ ] Tous les tests existants passent
- [ ] Nouveaux tests ajoutés
- [ ] Tests manuels effectués

## Vérification
- [ ] Le code suit le style du projet
- [ ] La documentation est mise à jour
- [ ] Aucune dépendance nouvelle requise (ou documentée)

## Issues liées
[Lien vers les issues liées]
```

---

## 🎁 Reconnaissance

Toutes les contributions sont appréciées et reconnues !

### Types de contributions reconnues

- **Code** : Nouveaux algorithmes, correctifs, optimisations
- **Documentation** : Amélioration des docs, exemples, tutoriels
- **Tests** : Ajout de tests, correction de bugs dans les tests
- **Review** : Review de PRs, suggestions d'amélioration
- **Reporting** : Signalement de bugs, suggestions de fonctionnalités
- **Autres** : Traduction, design, etc.

### Comment être reconnu

1. **Contributeurs** : Tous les contributeurs sont listés dans le fichier CONTRIBUTORS.md
2. **Changelog** : Les contributions majeures sont mentionnées dans le CHANGELOG.md
3. **GitHub** : Votre nom apparaît dans l'historique des commits

---

## 📜 Licence

En contribuant à ce projet, vous acceptez que vos contributions soient licenciées sous la même licence que le projet (MIT).

---

## 🙏 Remerciements

Merci à tous ceux qui contribuent à rendre AOT_biomaps meilleur !

Votre contribution, quelle que soit sa taille, est précieuse pour la communauté.

---

**Retour à la [documentation principale](README.md)**
