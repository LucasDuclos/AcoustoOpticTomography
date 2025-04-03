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

    print(f"x_range : {x_range.shape}")
    print(f"y_range : {y_range.shape}")
    print(f"z_range : {z_range.shape}")
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
    print(f"EnveloppeField data :  {EnveloppeField.shape}")
    print(f"reorganized data :  {reorganized_data.shape}")
    for y in range(reorganized_data.shape[1]):
        for z in range(reorganized_data.shape[2]):
            EnveloppeField[:, y, z, :] = np.abs(hilbert(reorganized_data[:, y, z, :], axis=1))

    return EnveloppeField.T, x_range, y_range, z_range

def generate_2Dacoustic_field_KWAVE(folderPathBase,depth_end, angle_deg, active_listString, c0=1540, num_elements = 192, num_cycles = 4, element_width = 0.2/1000, depth_start = 0, f_US = 180e6, f_aq = 10e6, IsSaving=True):
    active_listbin = ''.join(f"{int(active_listString[i:i+2], 16):08b}" for i in range(0, len(active_listString), 2))
    active_list = np.array([int(char) for char in active_listbin])
    print(active_list.shape)
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
    

def generate_3Dacoustic_field_KWAVE(folderPathBase,depth_end, angle_deg, active_listString, c0=1540, num_elements = 192, num_cycles = 4, element_width = 0.2/1000, element_height = 6/1000, depth_start = 0, f_US = 180e6, f_aq = 10e6, IsSaving=True):

    active_listbin = ''.join(f"{int(active_listString[i:i+2], 16):08b}" for i in range(0, len(active_listString), 2))
    active_list = np.array([int(char) for char in active_listbin])
    print(active_list.shape)
    # Grille
    probeWidth = num_elements * element_width
    Xrange = [-20 / 1000, 20 / 1000]  # Plage en X en mètres
    Yrange = [-element_height * 5 / 2, element_height * 5 / 2]  # Plage en Y en mètres
    Zrange = [depth_start, depth_end]  # Plage en Z en mètres

    t0 = floor(Zrange[0]/f_US)
    tmax = ceil((depth_end -depth_start + probeWidth*sin(radians(angle_deg)))/(c0*cos(radians(angle_deg)))*f_US)

    dx = element_width
    dz = dx
    dy = dx

    Nx = ceil((Xrange[1] - Xrange[0]) / dx)
    Ny = 4 * ceil((Yrange[1] - Yrange[0]) / element_height)
    Nz = ceil((Zrange[1] - Zrange[0]) / dz)
    Nt = tmax - t0 + 1

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
    print("Active List : ",active_listString)

    kgrid = kWaveGrid([Nx, Ny, Nz], [element_width, element_height, dx])
    kgrid.setTime(Nt = Nt, dt = 1/f_US)

    inputFileName = os.path.join(folderPathBase,"/KwaveIN.h5")
    outputFileName = os.path.join(folderPathBase,"/KwaveOUT.h5")

    # Définir le medium
    # medium = kWaveMedium(sound_speed=1540, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)
    medium = kWaveMedium(sound_speed=c0)
    
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
            x_pos = i  # Position des éléments sur l'axe X
            source.p_mask[x_pos, Ny // 2, 0] = 1  # Position dans le plan XZ

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
