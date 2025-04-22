import numpy as np

def gaussian_beam(x, z, center, w0 = 9/1000):
    """
    Génère un faisceau laser gaussien dans le plan XZ.

    Args:
        x, z (numpy array): Coordonnées X et Z de la grille.
        w0 (float): Waist du faisceau (demi-largeur à 1/e^2).
        center (tuple): Centre du faisceau (x0, z0).

    Returns:
        numpy array: Intensité du faisceau gaussien.
    """
    x0, z0 = center
    X, Z = np.meshgrid(x, z, indexing='ij')  # Crée une grille 2D
    return np.exp(-2 * ((X - x0)**2 + (Z - z0)**2) / w0**2)

def add_absorbers(intensity, x, z, absorbers):
    """
    Ajoute des absorbeurs circulaires avec absorption gaussienne au faisceau optique dans le plan XZ.

    Args:
        intensity (numpy array): Intensité du faisceau.
        x, z (numpy array): Coordonnées X et Z de la grille.
        absorbers (list of dict): Liste des absorbeurs avec `center` et `radius`.

    Returns:
        numpy array: Intensité modifiée avec absorbeurs.
    """
    # Création de la grille 2D pour X et Z
    X, Z = np.meshgrid(x, z, indexing='ij')
    
    # Copie de l'intensité pour éviter de modifier l'original
    modified_intensity = np.copy(intensity)

    for absorber in absorbers:
        center = absorber['center']  # Coordonnées (x_c, z_c)
        size = absorber['radius']      # Taille caractéristique (sigma de la gaussienne)
        absorption_strength = absorber.get('strength', 0.8)  # Coefficient d'absorption par défaut
        
        # Calcul de la distance au centre de l'absorbeur
        distance_squared = (X - center[0])**2 + (Z - center[1])**2
        
        # Appliquer la formule d'absorption gaussienne
        gaussian_absorption = -absorption_strength * np.exp(-distance_squared / size**2)
        
        # Ajouter l'absorption gaussienne à l'intensité
        modified_intensity += gaussian_absorption
    
    # Assurez-vous que l'intensité reste positive
    modified_intensity = np.clip(modified_intensity, 0, None)
    
    return modified_intensity
