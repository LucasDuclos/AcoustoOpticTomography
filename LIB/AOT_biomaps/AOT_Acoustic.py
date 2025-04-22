import scipy.io
import numpy as np
import h5py
from scipy.signal import hilbert
from math import ceil, sin, cos, radians, floor
import os
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.utils.signals import tone_burst
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
import matplotlib.pyplot as plt

def hex_to_binary_array(hex_string):
    # Convertir chaque caractère hexadécimal en 4 bits binaires
    binary_string = ''.join(f"{int(char, 16):04b}" for char in hex_string)
    # Convertir la chaîne binaire en un tableau NumPy de 0 et 1
    return np.array(list(binary_string), dtype=int)

def apply_delay(signal, angle_rad, kgrid_dt, is_positive, num_elements = 192, element_width = 0.2/1000, c0 = 1540):
    """
    Applique un retard temporel au signal pour chaque élément du transducteur.

    Args:
        signal (ndarray): Le signal acoustique initial.
        num_elements (int): Nombre total d'éléments.
        element_width (float): Largeur de chaque élément du transducteur.
        c0 (float): Vitesse du son dans le milieu (m/s).
        angle_rad (float): Angle d'inclinaison en radians.
        kgrid_dt (float): Pas de temps du kgrid.
        is_positive (bool): Indique si l'angle est positif ou négatif.

    Returns:
        ndarray: Tableau des signaux retardés.
    """
    delays = np.zeros(num_elements)
    for i in range(num_elements):
        delays[i] = (i * element_width * np.tan(angle_rad)) / c0  # Retard en secondes


    delay_samples = np.round(delays / kgrid_dt).astype(int)
    max_delay = np.max(np.abs(delay_samples))
    plt.figure(figsize=(12, 6))

    # Premier sous-graphique
    plt.figure()
    plt.plot(delays)
    plt.ylim([0, 10e-6])
    plt.title('Delays')
    plt.xlabel('Element')
    plt.ylabel('Delay (s)')

    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
    delayed_signals = np.zeros((num_elements, len(signal) + max_delay))

    for i in range(num_elements):
        shift = delay_samples[i]
        if is_positive:
            delayed_signals[i, shift:shift + len(signal)] = signal  # Décalage à droite
        else:
            delayed_signals[i, max_delay - shift:max_delay - shift + len(signal)] = signal  # Décalage à gauche

    return delayed_signals
    
def getActiveListBin(path):
    base_name = os.path.basename(path)
    file = os.path.splitext(base_name)[0]
    start = file.index('_') + 1
    end = file.index('_', start)
    hexa = file[start:end]
    return hex_to_binary_array(hexa)

def getActiveListHexa(path):
    base_name = os.path.basename(path)
    file = os.path.splitext(base_name)[0]
    start = file.index('_') + 1
    end = file.index('_', start)
    return file[start:end]

def getAngle(path):
    base_name = os.path.basename(path)
    file = os.path.splitext(base_name)[0]
    angle_str = file[-3:]
    if angle_str[0] == '0':
        sign = 1
    else:
        sign = -1
    return sign * int(angle_str[1:])

def load_fieldHYDRO_XZ(file_path_h5, param_path_mat):    

    # Charger les fichiers .mat
    param = scipy.io.loadmat(param_path_mat)

    # Charger les paramètres
    x_test = param['x'].flatten()
    z_test = param['z'].flatten()

    x_range = np.arange(-23,21.2,0.2)
    z_range = np.arange(0,37.2,0.2)
    X, Z = np.meshgrid(x_range, z_range)

    # Charger le fichier .h5
    with h5py.File(file_path_h5, 'r') as file:
        data = file['data'][:]

    # Initialiser une matrice pour stocker les données acoustiques
    acoustic_field = np.zeros((len(z_range), len(x_range), data.shape[1]))

    # Remplir la grille avec les données acoustiques
    index = 0
    for i in range(len(z_range)):
        if i % 2 == 0:
            # Parcours de gauche à droite
            for j in range(len(x_range)):
                acoustic_field[i, j, :] = data[index]
                index += 1
        else:
            # Parcours de droite à gauche
            for j in range(len(x_range) - 1, -1, -1):
                acoustic_field[i, j, :] = data[index]
                index += 1

     # Calculer l'enveloppe analytique
    envelope = np.abs(hilbert(acoustic_field, axis=2))
    # Réorganiser le tableau pour avoir la forme (Times, Z, X)
    envelope_transposed = np.transpose(envelope, (2, 0, 1))
    return envelope_transposed

def load_fieldHYDRO_YZ(file_path_h5, param_path_mat):
    # Load parameters from the .mat file
    param = scipy.io.loadmat(param_path_mat)

    # Extract the ranges for y and z
    y_range = param['y'].flatten()
    z_range = param['z'].flatten()

    # Load the data from the .h5 file
    with h5py.File(file_path_h5, 'r') as file:
        data = file['data'][:]

    # Calculate the number of scans
    Ny = len(y_range)
    Nz = len(z_range)
    Nscans = Ny * Nz

    # Create the scan positions
    positions_y = []
    positions_z = []

    for i in range(Nz):
        if i % 2 == 0:
            # Traverse top to bottom for even rows
            positions_y.extend(y_range)
        else:
            # Traverse bottom to top for odd rows
            positions_y.extend(y_range[::-1])
        positions_z.extend([z_range[i]] * Ny)

    Positions = np.column_stack((positions_y, positions_z))

    # Initialize a matrix to store the reorganized data
    reorganized_data = np.zeros((Ny, Nz, data.shape[1]))

    # Reorganize the data according to the scan positions
    for index, (j, k) in enumerate(Positions):
        y_idx = np.where(y_range == j)[0][0]
        z_idx = np.where(z_range == k)[0][0]
        reorganized_data[y_idx, z_idx, :] = data[index, :]

    # Calculer l'enveloppe analytique
    envelope = np.abs(hilbert(reorganized_data, axis=2))
    # Réorganiser le tableau pour avoir la forme (Times, Z, Y)
    envelope_transposed = np.transpose(envelope, (2, 0, 1))
    return envelope_transposed, y_range, z_range

def load_fieldHydro_XYZ(file_path_h5, param_path_mat):
    # Load parameters from the .mat file
    param = scipy.io.loadmat(param_path_mat)

    # Extract the ranges for x, y, and z
    x_range = param['x'].flatten()
    y_range = param['y'].flatten()
    z_range = param['z'].flatten()

    # Create a meshgrid for x, y, and z
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Load the data from the .h5 file
    with h5py.File(file_path_h5, 'r') as file:
        data = file['data'][:]

    # Calculate the number of scans
    Nx = len(x_range)
    Ny = len(y_range)
    Nz = len(z_range)
    Nscans = Nx * Ny * Nz

    # Create the scan positions
    if Ny % 2 == 0:
        X = np.tile(np.concatenate([x_range[:, np.newaxis], x_range[::-1, np.newaxis]]), (Ny // 2, 1))
        Y = np.repeat(y_range, Nx)
    else:
        X = np.concatenate([x_range[:, np.newaxis], np.tile(np.concatenate([x_range[::-1, np.newaxis], x_range[:, np.newaxis]]), ((Ny - 1) // 2, 1))])
        Y = np.repeat(y_range, Nx)

    XY = np.column_stack((X.flatten(), Y))

    if Nz % 2 == 0:
        XYZ = np.tile(np.concatenate([XY, np.flipud(XY)]), (Nz // 2, 1))
        Z = np.repeat(z_range, Nx * Ny)
    else:
        XYZ = np.concatenate([XY, np.tile(np.concatenate([np.flipud(XY), XY]), ((Nz - 1) // 2, 1))])
        Z = np.repeat(z_range, Nx * Ny)

    Positions = np.column_stack((XYZ, Z))

    # Initialize a matrix to store the reorganized data
    reorganized_data = np.zeros((Nx, Ny, Nz, data.shape[1]))

    # Reorganize the data according to the scan positions
    for index, (i, j, k) in enumerate(Positions):
        x_idx = np.where(x_range == i)[0][0]
        y_idx = np.where(y_range == j)[0][0]
        z_idx = np.where(z_range == k)[0][0]
        reorganized_data[x_idx, y_idx, z_idx, :] = data[index, :]
    
    EnveloppeField = np.zeros_like(reorganized_data)

    for y in range(reorganized_data.shape[1]):
        for z in range(reorganized_data.shape[2]):
            EnveloppeField[:, y, z, :] = np.abs(hilbert(reorganized_data[:, y, z, :], axis=1))

    return EnveloppeField.T, x_range, y_range, z_range

def generate_2Dacoustic_field_KWAVE(folderPathBase,depth_end, angle_deg, active_listString, c0=1540, num_elements = 192, num_cycles = 4, element_width = 0.2/1000, depth_start = 0, f_US = 180e6, f_aq = 10e6, IsSaving=True):
    active_listbin = ''.join(f"{int(active_listString[i:i+2], 16):08b}" for i in range(0, len(active_listString), 2))
    active_list = np.array([int(char) for char in active_listbin])
    # Grille
    probeWidth = num_elements * element_width
    Xrange = [-20 / 1000, 20 / 1000]  # Plage en X en mètres
    Zrange = [depth_start, depth_end ]  # Z range in meters for 289 time samples and 10° max

    t0 = floor(Zrange[0]/f_US)
    tmax = ceil((depth_end -depth_start + probeWidth*sin(radians(angle_deg)))/(c0*cos(radians(angle_deg)))*f_US)

    Nx = ceil((Xrange[1] - Xrange[0]) / element_width)
    Nz = ceil((Zrange[1] - Zrange[0]) / element_width)
    Nt = tmax - t0 + 1

    dx = element_width
    dz = dx

    # Print the results
    print("Xrange:", Xrange)
    print("Zrange:", Zrange)
    print("Nx:", Nx)
    print("Nz:", Nz)
    print("dx:",dx)
    print("dz:",dz)
    print("Angles : ",angle_deg)
    print("Active List : ",active_listString)

    kgrid = kWaveGrid([Nx, Nz], [dx, dz])
    kgrid.setTime(Nt = Nt, dt = 1/f_US)

    inputFileName = os.path.join(folderPathBase,"/KwaveIN.h5")
    outputFileName = os.path.join(folderPathBase,"/KwaveOUT.h5")

    # Définir le medium
    # medium = kWaveMedium(sound_speed=1540, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)
    medium = kWaveMedium(sound_speed=c0)
    
    acoustic_field = np.zeros((kgrid.Nt, Nz, Nx))
    
    # Génération du signal de base
    signal = tone_burst(1.5 / kgrid.dt, f_US, num_cycles).squeeze() # * 1.5 pour faire comme dans Field2

    # Masque de la sonde : alignée dans le plan XZ
    source = kSource()
    source.p_mask = np.zeros((Nx, Nz))  # Crée une grille vide pour le masque de la source
 
    # Placement des transducteurs actifs dans le masque
    for i in range(num_elements):
        if active_list[i] == 1:  # Vérifiez si l'élément est actif
            x_pos = i  # Position des éléments sur l'axe X
            source.p_mask[x_pos, 0] = 1  # Position dans le plan XZ

    source.p_mask = source.p_mask.astype(int)  # Conversion en entier
    # print("Number of active elements in p_mask:", np.sum(source.p_mask))
    is_positive_angle = angle_deg >= 0
    # Inclinaison de la sonde (en degrés)
    angle_rad = np.radians(abs(angle_deg))  # Convertir en radians

    delayed_signals = apply_delay(signal, num_elements, element_width, c0, angle_rad, kgrid.dt, is_positive_angle)

    # Filtrer les signaux pour correspondre aux éléments actifs
    delayed_signals_active = delayed_signals[active_list == 1, :]
    source.p = delayed_signals_active  # Transposer pour que chaque colonne corresponde à un élément actif

    # === Définir les capteurs pour observer les champs acoustiques ===
    sensor = kSensor()
    sensor.mask = np.ones((Nx, Nz))  # Capteur couvrant tout le domaine

    # === Options de simulation ===
    simulation_options = SimulationOptions(
        data_cast="single",
        pml_inside=False,  # Empêche le PML d'être ajouté à l'intérieur de la grille
        pml_x_size=20,      # Taille de la PML sur l'axe X
        pml_z_size=20,       # Taille de la PML sur l'axe Z  
        use_sg=False,           # Pas de Staggered Grid         
        save_to_disk=True,
        input_filename=inputFileName,
        output_filename=outputFileName,
        data_path=folderPathBase)

    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)  # True si GPU disponible

    # === Lancer la simulation ===
    print("Lancement de la simulation...")
    sensor_data = kspaceFirstOrder2D(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    print("Simulation terminée avec succès.")

    acoustic_field = sensor_data['p'].reshape(kgrid.Nt,Nz, Nx)
    print("Calcul de l'enveloppe acoustique...")
    
    for y in range(acoustic_field.shape[2]):
        acoustic_field[:, :, y, :]= np.abs(hilbert(acoustic_field[:, :, y, :], axis=0))

    acoustic_envelope_squared = np.sum(acoustic_field, axis=2)**2

    if f_US != f_aq:
        downsample_factor = int(f_US / f_aq)
    else:
        downsample_factor = 1    

    acoustic_field_ToSave = acoustic_envelope_squared[::downsample_factor, :, :]
    
    if IsSaving:
        print("Saving...")
        save_field(acoustic_field_ToSave, num_elements, active_list, angle_deg, folderPathBase, dx, f_aq,(len(signal)-1)*kgrid.dt)
    return acoustic_field_ToSave
    
def generate_3Dacoustic_field_KWAVE(folderPathBase,depth_end, angle_deg, active_list_hex, c0=1540, num_elements = 192, num_cycles = 4, element_width = 0.2/1000, element_height = 6/1000, depth_start = 0, f_US = 180e6, f_aq = 180e6, IsSaving=True):
    print((active_list_hex))
    active_listbin = ''.join(f"{int(active_list_hex[i:i+2], 16):08b}" for i in range(0, len(active_list_hex), 2))
    active_list = np.array([int(char) for char in active_listbin])
    angle_sign = '1' if angle_deg < 0 else '0'
    formatted_angle = f"{angle_sign}{abs(angle_deg):02d}"
    file_name = f"KWAVE_{active_list_hex}_{formatted_angle}"
    hdr_path = os.path.join(folderPathBase, file_name + ".hdr")
    print(active_list.shape)
    
    # Grille
    probeWidth = num_elements * element_width
    Xrange = [-24 / 1000, 24 / 1000]  # Plage en X en mètres
    Yrange = [-element_height * 5 / 2, element_height * 5 / 2]  # Plage en Y en mètres
    Zrange = [depth_start, depth_end]  # Plage en Z en mètres

    dx = element_width
    dz = dx
    dy = dx

    t0 = floor(Zrange[0] / f_aq)
    tmax = ceil((depth_end - depth_start + probeWidth * sin(radians(abs(angle_deg)))) / (c0 * cos(radians(abs(angle_deg)))) * f_aq)

    Nx = ceil((Xrange[1] - Xrange[0]) / element_width)
    Ny = 4 * ceil((Yrange[1] - Yrange[0]) / element_height)
    Nz = ceil((Zrange[1] - Zrange[0]) / element_width)
    Nt = int(np.round(1.5*(tmax - t0)))

    # Print the results
    print("Xrange:", Xrange)
    print("Yrange:", Yrange)
    print("Zrange:", Zrange)
    print("Nx:", Nx)
    print("Ny:", Ny)
    print("Nz:", Nz)
    print("dx:",dx)
    print("dy:",dy)
    print("dz:",dz)
    print("Angles : ",angle_deg)
    print("Active List : ",active_list_hex)
    print("Nt:", Nt)

    # Initialisation de la grille et du milieu
    kgrid = kWaveGrid([Nx, Ny, Nz], [element_width, element_height, dx])
    kgrid.setTime(Nt=Nt, dt=1/f_aq)
    medium = kWaveMedium(sound_speed=c0)

    inputFileName = os.path.join(folderPathBase,"KwaveIN.h5")
    outputFileName = os.path.join(folderPathBase,"KwaveOUT.h5")

    
    acoustic_field = np.zeros((kgrid.Nt, Nz, Ny, Nx))
    acoustic_envelope_squared = np.zeros((kgrid.Nt, Nz, Nx))
    
    # Génération du signal de base
    signal = tone_burst(1.5 / kgrid.dt, f_US, num_cycles).squeeze() # * 1.5 pour faire comme dans Field2

    # Masque de la sonde : alignée dans le plan XZ
    source = kSource()
    source.p_mask = np.zeros((Nx, Ny, Nz))  # Crée une grille vide pour le masque de la source

    stringList = ''.join(map(str, active_list))
    print(stringList)
 
    # Placement des transducteurs actifs dans le masque
    for i in range(num_elements):
        if active_list[i] == 1:  # Vérifiez si l'élément est actif
            x_pos = i+Nx//2 - num_elements//2 # Position des éléments sur l'axe X
            source.p_mask[x_pos, Ny // 2, 0] = 1  # Position dans le plan XZ

    source.p_mask = source.p_mask.astype(int)  # Conversion en entier
    # print("Number of active elements in p_mask:", np.sum(source.p_mask))
    is_positive_angle = angle_deg >= 0
    # Inclinaison de la sonde (en degrés)
    angle_rad = np.radians(abs(angle_deg))  # Convertir en radians

    delayed_signals = apply_delay(signal, angle_rad, kgrid.dt, is_positive_angle, num_elements, element_width, c0)

    # Filtrer les signaux pour correspondre aux éléments actifs
    delayed_signals_active = delayed_signals[active_list == 1, :]
    source.p = delayed_signals_active  # Transposer pour que chaque colonne corresponde à un élément actif

    # === Définir les capteurs pour observer les champs acoustiques ===
    sensor = kSensor()
    sensor.mask = np.ones((Nx, Ny, Nz))  # Capteur couvrant tout le domaine

    # === Options de simulation ===
    simulation_options = SimulationOptions(
        data_cast="single",
        pml_inside=False,  # Empêche le PML d'être ajouté à l'intérieur de la grille
        pml_x_size=20,      # Taille de la PML sur l'axe X
        pml_y_size=2,      # Taille de la PML sur l'axe Y
        pml_z_size=20,       # Taille de la PML sur l'axe Z  
        use_sg=False,           # Pas de Staggered Grid         
        save_to_disk=True,
        input_filename=inputFileName,
        output_filename=outputFileName,
        data_path=folderPathBase)

    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)  # True si GPU disponible

    # === Lancer la simulation ===
    print("Lancement de la simulation...")
    sensor_data = kspaceFirstOrder3D(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    print("Simulation terminée avec succès.")
    acoustic_field = sensor_data['p'].reshape(kgrid.Nt,Nz, Ny, Nx)
    
    downsample_factor = int(180/25)

    acoustic_fieldSampled = acoustic_field[::downsample_factor, :,:, :]

    EnveloppeField = np.zeros_like(acoustic_fieldSampled)
    for y in range(acoustic_field.shape[2]):
        for z in range(acoustic_field.shape[1]):
            EnveloppeField[:, z, y, :] = np.abs(hilbert(acoustic_fieldSampled[:, z, y, :], axis=1))
  
    print(f"acoustic_envelope_squared : {EnveloppeField.shape}")
    sliceEnvelopeField = np.sum(EnveloppeField, axis=2)**2
    
    if IsSaving:
        print("Saving...")
        save_field(sliceEnvelopeField, hdr_path)
    return sliceEnvelopeField

def save_AOsignal(AOsignal,listHDRpath, save_directory,fs_aq=25e6, num_elements=192):
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

    # Calcul des angles (en degrés) si nécessaire

    nScan = AOsignal.shape[1]  # Nombre de scans ou d'événements

    # **1. Sauvegarde du fichier .cdf**
    with open(cdf_location, "wb") as fileID:
        for j in range(nScan):
            file = listHDRpath[j]
            active_list = getActiveListBin(file)
            angle = getAngle(file)
             # Écrire les identifiants hexadécimaux
            active_list_str = ''.join(map(str, active_list)) 

            nb_padded_zeros = (4 - len(active_list_str) % 4) % 4  # Calcul du nombre de 0 nécessaires
            active_list_str += '0' * nb_padded_zeros  # Ajout des zéros à la fin de la chaîne

            # Regrouper par paquets de 4 bits et convertir chaque paquet en hexadécimal
            active_list_hex = ''.join([hex(int(active_list_str[i:i+4], 2))[2:] for i in range(0, len(active_list_str), 4)])

            for i in range(0, len(active_list_hex), 2):  # Chaque 2 caractères hex représentent 1 octet
                byte_value = int(active_list_hex[i:i + 2], 16)  # Convertit l'hexadécimal en entier
                fileID.write(byte_value.to_bytes(1, byteorder='big'))  # Écriture en big endian
        
            fileID.write(np.int8(angle).tobytes())
            
            # Écrire le signal AO correspondant (times x 1) en single (float32)
            fileID.write(AOsignal[:, j].astype(np.float32).tobytes())

   # **2. Sauvegarde du fichier .cdh**
    header_content = (
        f"Data filename: AOSignals.cdf\n"
        f"Number of events: {nScan}\n"
        f"Number of acquisitions per event: {AOsignal.shape[1]}\n"
        f"Start time (s): 0\n"
        f"Duration (s): 1\n"
        f"Acquisition frequency (Hz): {fs_aq}\n"
        f"Data mode: histogram\n"
        f"Data type: AOT\n"
        f"Number of US transducers: {num_elements}"
    )
    with open(cdh_location, "w") as fileID:
        fileID.write(header_content)

    with open(info_location, "w") as fileID:
        for path in listHDRpath:
            fileID.write(path + "\n")

def save_field(acoustic_field, filePath, f0=6e6, num_elements =192, dx = 0.2/1000):
    """
    Fonction Python qui reproduit la logique de la méthode SaveField du code MATLAB.

    Paramètres :
    - obj : Objet contenant les paramètres (comme obj.param en MATLAB).
    - acoustic_field : Données du champ acoustique (MySimulationBox.Field en MATLAB).
    - num_elements : Nombre total d'éléments de la sonde.
    - structuration : Structure d'activation des transducteurs.
    - folderPath : Chemin où les fichiers .img et .hdr seront enregistrés.
    """
    t_ex = 1/f0

    active_list_hex = getActiveListHexa(filePath)
    active_list_bin = ''.join(map(str,getActiveListBin(filePath)))

    angle = getAngle(filePath)
    angle_sign = '1' if angle < 0 else '0'
    formatted_angle = f"{angle_sign}{abs(angle):02d}"

    # 4. Définir les noms de fichiers (img et hdr)
    file_name = f"field_{active_list_hex}_{formatted_angle}"

    img_path = os.path.join(Path(filePath).parent , file_name + ".img")
    hdr_path = os.path.join(Path(filePath).parent , file_name + ".hdr")
    

    # === 3. Sauvegarder le champ acoustique dans le fichier .img ===
    with open(img_path, "wb") as f_img:
        acoustic_field.astype('float32').tofile(f_img)  # Sauvegarde au format float32 (équivalent à "single" en MATLAB)
    
    # === 4. Création du contenu du fichier .hdr ===
    x_range = [0, acoustic_field.shape[2] * dx]
    z_range = [0, acoustic_field.shape[1] * dx]
    x_pixel_size = dx  # En mm/pixel
    z_pixel_size = dx  # En mm/pixel
    time_pixel_size = 1 / 25e6  # En s/pixel
    first_pixel_offset_x = x_range[0] * 1e3  # En mm
    first_pixel_offset_z = z_range[0] * 1e3  # En mm
    first_pixel_offset_t = 0

    # **Génération du headerFieldGlob**
    headerFieldGlob = (
        f"!INTERFILE :=\n"
        f"modality : AOT\n"
        f"voxels number transaxial: {acoustic_field.shape[2]}\n"
        f"voxels number transaxial 2: {acoustic_field.shape[1]}\n"
        f"voxels number axial: {1}\n"
        f"field of view transaxial: {(x_range[1] - x_range[0]) * 1000}\n"
        f"field of view transaxial 2: {(z_range[1] - z_range[0]) * 1000}\n"
        f"field of view axial: {1}\n"
    )

    # **Génération du header**
    header = (
        f"!INTERFILE :=\n"
        f"!imaging modality := AOT\n\n"
        f"!GENERAL DATA :=\n"
        f"!data offset in bytes := 0\n"
        f"!name of data file := system_matrix/{file_name}.img\n\n"
        f"!GENERAL IMAGE DATA\n"
        f"!total number of images := {acoustic_field.shape[0]}\n"
        f"imagedata byte order := LITTLEENDIAN\n"
        f"!number of frame groups := 1\n\n"
        f"!STATIC STUDY (General) :=\n"
        f"number of dimensions := 3\n"
        f"!matrix size [1] := {acoustic_field.shape[2]}\n"
        f"!matrix size [2] := {acoustic_field.shape[1]}\n"
        f"!matrix size [3] := {acoustic_field.shape[0]}\n"
        f"!number format := short float\n"
        f"!number of bytes per pixel := 4\n"
        f"scaling factor (mm/pixel) [1] := {x_pixel_size * 1000}\n"
        f"scaling factor (mm/pixel) [2] := {z_pixel_size * 1000}\n"
        f"scaling factor (s/pixel) [3] := {time_pixel_size}\n"
        f"first pixel offset (mm) [1] := {first_pixel_offset_x}\n"
        f"first pixel offset (mm) [2] := {first_pixel_offset_z}\n"
        f"first pixel offset (s) [3] := {first_pixel_offset_t}\n"
        f"data rescale offset := 0\n"
        f"data rescale slope := 1\n"
        f"quantification units := 1\n\n"
        f"!SPECIFIC PARAMETERS :=\n"
        f"angle (degree) := {angle}\n"
        f"activation list := {active_list_bin}\n"
        f"number of US transducers := {num_elements}\n"
        f"delay (s) := 0\n"
        f"us frequency (Hz) := {f0}\n"
        f"excitation duration (s) := {t_ex}\n"
        f"!END OF INTERFILE :=\n"
    )

    # === 5. Sauvegarder le fichier .hdr ===
    with open(hdr_path, "w") as f_hdr:
        f_hdr.write(header)

    with open(os.path.join(Path(filePath).parent ,"field.hdr"), "w") as f_hdr2:
        f_hdr2.write(headerFieldGlob)

def load_fieldKWAVE_XZ(hdr_path):
    """
    Lit un fichier Interfile (.hdr) et son fichier binaire (.img) pour reconstruire un champ acoustique.

    Paramètres :
    ------------
    - folderPathBase : dossier de base contenant les fichiers
    - hdr_path : chemin relatif du fichier .hdr depuis folderPathBase

    Retour :
    --------
    - field : tableau NumPy contenant le champ acoustique avec les dimensions réordonnées en (X, Z, time)
    - header : dictionnaire contenant les métadonnées du fichier .hdr
    """
    header = {}
    # Lecture du fichier .hdr
    with open(hdr_path, 'r') as f:
        for line in f:
            if ':=' in line:
                key, value = line.split(':=', 1)
                key = key.strip().lower().replace('!', '')
                value = value.strip()
                header[key] = value


    # Récupère le nom du fichier .img associé
    data_file = header.get('name of data file') or header.get('name of date file')
    if data_file is None:
        raise ValueError(f"Impossible de trouver le fichier de données associé au fichier header {hdr_path}")
    img_path = os.path.join(os.path.dirname(hdr_path),os.path.basename(data_file))

    # Détermine la taille du champ à partir des métadonnées
    shape = [int(header[f'matrix size [{i}]']) for i in range(1, 3) if f'matrix size [{i}]' in header]
    if not shape:
        raise ValueError("Impossible de déterminer la forme du champ acoustique à partir des métadonnées.")

    # Type de données
    data_type = header.get('number format', 'short float').lower()
    dtype_map = {
        'short float': np.float32,
        'float': np.float32,
        'int16': np.int16,
        'int32': np.int32,
        'uint16': np.uint16,
        'uint8': np.uint8
    }
    dtype = dtype_map.get(data_type)
    if dtype is None:
        raise ValueError(f"Type de données non pris en charge : {data_type}")

    # Ordre des octets (endianness)
    byte_order = header.get('imagedata byte order', 'LITTLEENDIAN').lower()
    endianess = '<' if 'little' in byte_order else '>'

    # Vérifie la taille réelle du fichier .img
    fileSize = os.path.getsize(img_path)
    timeDim = int(fileSize / (np.dtype(dtype).itemsize *np.prod(shape)))
        # if img_size != expected_size:
    #     raise ValueError(f"La taille du fichier img ({img_size} octets) ne correspond pas à la taille attendue ({expected_size} octets).")
    shape = [timeDim] + shape
    # Lecture des données binaires
    with open(img_path, 'rb') as f:
        data = np.fromfile(f, dtype=endianess + np.dtype(dtype).char)

    # Reshape les données en (time, Z, X)
    field = data.reshape(shape[::-1])  # NumPy interprète dans l'ordre C (inverse de MATLAB)



    # Applique les facteurs d'échelle si disponibles
    rescale_slope = float(header.get('data rescale slope', 1))
    rescale_offset = float(header.get('data rescale offset', 0))
    field = field * rescale_slope + rescale_offset

    return field
