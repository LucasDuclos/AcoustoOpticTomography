import os
import shutil
from .AOT_Acoustic import *
import numpy as np
import re
import time

def getAOsignal(laser, acoustic_envelope_squaredMATRIX):
    AOsignal = np.zeros((acoustic_envelope_squaredMATRIX.shape[3], acoustic_envelope_squaredMATRIX.shape[0]))
    for i in range (acoustic_envelope_squaredMATRIX.shape[3]):
        for t in range(acoustic_envelope_squaredMATRIX.shape[0]):
            interaction = laser * acoustic_envelope_squaredMATRIX[t, :, :, i]
            AOsignal[i, t] = np.sum(interaction)
    return AOsignal

def getSaveAOsignal(listFieldPathHDR, laser, acoustic_envelope_squaredMATRIX, save_directory):
    
    AOsignal = getAOsignal(laser, acoustic_envelope_squaredMATRIX)

    active_lists = []
    angles = []
    angle_pattern = re.compile(r"_([0-9]+)\.hdr$")
    for i in range(len(listFieldPathHDR)):
    # Extraire l'active list
        start = listFieldPathHDR[i].find("field_") + len("field_")
        end = listFieldPathHDR[i].find("_", start)
        active_list = listFieldPathHDR[i][start:end]
        active_lists.append(active_list)
        # Extraire l'angle
        angle_match = angle_pattern.search(listFieldPathHDR[i])
        if angle_match:
            angle_str = angle_match.group(1)
            sign = -1 if angle_str[0] == '1' else 1
            value = int(angle_str[1:])
            angle = sign * value
            angles.append(angle)


    headerFieldGlob = (
            f"!INTERFILE :=\n"
            f"modality : AOT\n"
            f"voxels number transaxial: 200\n"
            f"voxels number transaxial 2: 185\n"
            f"voxels number axial: {1}\n"
            f"field of view transaxial: 40.0\n"
            f"field of view transaxial 2: 37.0\n"
            f"field of view axial: {1}\n"
        )
    with open(save_directory + "/system_matrix/field.hdr", "w") as f_hdr2:
        f_hdr2.write(headerFieldGlob)

    activeListarray = np.zeros((len(active_lists), 192),dtype=int)
    for i in range(len(active_lists)):
        activeListarray[i] = hex_to_binary_array(active_lists[i])
    save_AOsignal(AOsignal, save_directory, angles, acoustic_envelope_squaredMATRIX.shape[3], activeListarray)

    return AOsignal


def load_AOMatrix(cdh_file):
    # Lire les métadonnées depuis le fichier .cdh
    with open(cdh_file, "r") as file:
        cdh_content = file.readlines()

    # Extraire les dimensions des données à partir des métadonnées
    n_events = int([line.split(":")[1].strip() for line in cdh_content if "Number of events" in line][0])
    n_acquisitions = int([line.split(":")[1].strip() for line in cdh_content if "Number of acquisitions per event" in line][0])

    # Initialiser la matrice pour stocker les données
    AOsignal_matrix = np.zeros((n_events, n_acquisitions), dtype=np.float32)

    # Lire les données binaires depuis le fichier .cdf
    with open(cdh_file.replace(".cdh", ".cdf"), "rb") as file:
        for event in range(n_events):
            # Lire et ignorer la chaîne hexadécimale (active_list)
            num_elements = int([line.split(":")[1].strip() for line in cdh_content if "Number of US transducers" in line][0])
            hex_length = (num_elements + 3) // 4  # Nombre de caractères hex nécessaires
            file.read(hex_length // 2)  # Ignorer la chaîne hexadécimale
            
            # Lire l'angle (int8)
            angle = int.from_bytes(file.read(1), byteorder='big', signed=True)
            
            # Lire le signal AO correspondant (float32)
            signal = np.frombuffer(file.read(n_acquisitions * 4), dtype=np.float32)  # 4 octets par float32
            
            # Stocker le signal dans la matrice
            AOsignal_matrix[event, :] = signal

    return AOsignal_matrix


def save_AOsignal(AOsignal, save_directory, angles, Nt, active_list):
    """
    Sauvegarde le signal AO au format .cdf et .cdh (comme dans le script MATLAB)
    
    :param AOsignal: np.ndarray de taille (times, angles) 
    :param save_directory: chemin de sauvegarde
    :param set_id: identifiant du set
    :param n_experiment: identifiant de l'expérience
    :param param: dictionnaire contenant les paramètres nécessaires (comme fs_aq, Nt, angles, etc.)
    """

    # Noms des fichiers de sortie
    cdf_location = os.path.join(save_directory, "AOSignals.cdf")
    cdh_location = os.path.join(save_directory, "AOSignals.cdh")
    info_location = os.path.join(save_directory, "info.txt")

    nScan = AOsignal.shape[1]  # Nombre de scans ou d'événements
    # **1. Sauvegarde du fichier .cdf**
    with open(cdf_location, "wb") as fileID:
        for j in range(nScan):
             # Écrire les identifiants hexadécimaux
            active_list_str = ''.join(map(str, active_list[j,:] )) 
            nb_padded_zeros = (4 - len(active_list_str) % 4) % 4  # Calcul du nombre de 0 nécessaires
            active_list_str += '0' * nb_padded_zeros  # Ajout des zéros à la fin de la chaîne

            # Regrouper par paquets de 4 bits et convertir chaque paquet en hexadécimal
            active_list_hex = ''.join([hex(int(active_list_str[i:i+4], 2))[2:] for i in range(0, len(active_list_str), 4)])
            for i in range(0, len(active_list_hex), 2):  # Chaque 2 caractères hex représentent 1 octet
                byte_value = int(active_list_hex[i:i + 2], 16)  # Convertit l'hexadécimal en entier
                fileID.write(byte_value.to_bytes(1, byteorder='big'))  # Écriture en big endian
        
            # Écrire l'angle correspondant en int8
            angle = angles[j] # Récupération de l'index d'angle correspondant
            fileID.write(np.int8(angle).tobytes())
            
            # Écrire le signal AO correspondant (times x 1) en single (float32)
            fileID.write(AOsignal[:,j].astype(np.float32).tobytes())

   # **2. Sauvegarde du fichier .cdh**
    header_content = (
        f"Data filename: AOSignals.cdf\n"
        f"Number of events: {nScan}\n"
        f"Number of acquisitions per event: {Nt}\n"
        f"Start time (s): 0\n"
        f"Duration (s): 1\n"
        f"Acquisition frequency (Hz): 10000000.0\n"
        f"Data mode: histogram\n"
        f"Data type: AOT\n"
        f"Number of US transducers: 192"
    )
    with open(cdh_location, "w") as fileID:
        fileID.write(header_content)

    # **3. Sauvegarde du fichier info.txt**
    info_content = (
        f"Angles: {' '.join(map(str, angles))}\n"
        f"Decimations: ...\n"
        f"Number of shifts: 1\n"
        f"Fundamental: 1"
    )
    with open(info_location, "w") as fileID:
        fileID.write(info_content)

    print(f"Fichiers .cdf, .cdh et info.txt sauvegardés dans {save_directory}")

