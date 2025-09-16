#!/bin/bash
# Charge le secret dans l'environnement
export PYPI_API_TOKEN="$(gh secret get PYPI_API_TOKEN)"
# Exécute le script PowerShell
pwsh maj.ps1
