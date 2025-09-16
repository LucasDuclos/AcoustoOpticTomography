# Nettoyage des anciens builds
Remove-Item -Recurse -Force build, dist, AOT_biomaps.egg-info -ErrorAction SilentlyContinue

# Création du fichier .pypirc avec le token (récupéré depuis l'environnement)
$pypircContent = @"
[distutils]
index-servers =
  pypi

[pypi]
username = __token__
password = $env:PYPI_API_TOKEN
"@
$pypircContent | Out-File -FilePath ~/.pypirc -Encoding utf8

# Build et upload
python setup.py sdist bdist_wheel
twine upload dist/*

# Mise à jour locale
pip install --upgrade AOT-biomaps
pip install --upgrade AOT-biomaps
