def load_field(hdr_path):
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
