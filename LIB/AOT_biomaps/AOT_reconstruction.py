import subprocess
import os
import numpy as np

def makeRecon(AO_path, sMatrixDir,imageDir,reconExe):

    # Check if the input file exists
    if not os.path.exists(AO_path):
        print(f"Error: no input file {AO_path}")
        exit(1)

    # Check if the system matrix directory exists
    if not os.path.exists(sMatrixDir):
        print(f"Error: no system matrix directory {sMatrixDir}")
        exit(2)

    # Create the output directory if it does not exist
    os.makedirs(imageDir, exist_ok=True)

    opti = "MLEM"
    penalty = ""
    iteration = "100:10"

    cmd = (
        f"{reconExe} -df {AO_path} -opti {opti} {penalty} "
        f"-it {iteration} -proj matrix -dout {imageDir} -th 24 -vb 5 -proj-comp 1 -ignore-scanner "
        f"-data-type AOT -ignore-corr cali,fdur -system-matrix {sMatrixDir}"
    )
    result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    print(result.stdout)

def read_recon(hdr_path):
    """
    Lit un fichier Interfile (.hdr) et son fichier binaire (.img) pour reconstruire une image comme le fait Vinci.
    
    Paramètres :
    ------------
    - hdr_path : chemin complet du fichier .hdr
    
    Retour :
    --------
    - image : tableau NumPy contenant l'image
    - header : dictionnaire contenant les métadonnées du fichier .hdr
    """
    header = {}
    with open(hdr_path, 'r') as f:
        for line in f:
            if ':=' in line:
                key, value = line.split(':=', 1)  # s'assurer qu'on ne coupe que la première occurrence de ':='
                key = key.strip().lower().replace('!', '')  # Nettoyage des caractères
                value = value.strip()
                header[key] = value
    
    # 📘 Obtenez le nom du fichier de données associé (le .img)
    data_file = header.get('name of data file')
    if data_file is None:
        raise ValueError(f"Impossible de trouver le fichier de données associé au fichier header {hdr_path}")
    
    img_path = os.path.join(os.path.dirname(hdr_path), data_file)
    
    # 📘 Récupérer la taille de l'image à partir des métadonnées
    shape = [int(header[f'matrix size [{i}]']) for i in range(1, 4) if f'matrix size [{i}]' in header]
    if shape and shape[-1] == 1:  # Si la 3e dimension est 1, on la supprime
        shape = shape[:-1]  # On garde (192, 240) par exemple
    
    if not shape:
        raise ValueError("Impossible de déterminer la forme de l'image à partir des métadonnées.")
    
    # 📘 Déterminez le type de données à utiliser
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
    
    # 📘 Ordre des octets (endianness)
    byte_order = header.get('imagedata byte order', 'LITTLEENDIAN').lower()
    endianess = '<' if 'little' in byte_order else '>'
    
    # 📘 Vérifie la taille réelle du fichier .img
    img_size = os.path.getsize(img_path)
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize
    
    if img_size != expected_size:
        raise ValueError(f"La taille du fichier img ({img_size} octets) ne correspond pas à la taille attendue ({expected_size} octets).")
    
    # 📘 Lire les données binaires et les reformater
    with open(img_path, 'rb') as f:
        data = np.fromfile(f, dtype=endianess + np.dtype(dtype).char)
    
    image =  data.reshape(shape[::-1]) 
    
    # 📘 Rescale l'image si nécessaire
    rescale_slope = float(header.get('data rescale slope', 1))
    rescale_offset = float(header.get('data rescale offset', 0))
    image = image * rescale_slope + rescale_offset
    
    return image.T