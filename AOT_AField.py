def load_fieldKWAVE(hdr_path):
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
    shape = [int(header[f'matrix size [{i}]']) for i in range(1, 4) if f'matrix size [{i}]' in header]
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
    img_size = os.path.getsize(img_path)
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize
    if img_size != expected_size:
        raise ValueError(f"La taille du fichier img ({img_size} octets) ne correspond pas à la taille attendue ({expected_size} octets).")

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

def load_fieldHYDRO_XYZ(file_path_h5, param_path_mat):
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
