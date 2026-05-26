import os
import sys
import platform
import subprocess
import zipfile
import shutil
import tempfile
import urllib.request
from pathlib import Path

def ensure_system_dependencies():
    """
    Checks and installs system dependencies (libsz2, libaec0, libfftw3, szip)
    in user-space (~/.local) without requiring sudo privileges.
    """
    system = platform.system()
    user_home = Path.home()
    
    if system == 'Linux':
        user_lib_dir = user_home / ".local" / "lib"
        user_lib_dir.mkdir(parents=True, exist_ok=True)
        _ensure_linux_dependencies(user_lib_dir)
        
        # Dynamic injection into LD_LIBRARY_PATH so the OS and k-Wave find the libs
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if str(user_lib_dir) not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{user_lib_dir}:{current_ld_path}"
            
    elif system == 'Windows':
        user_lib_dir = user_home / ".local" / "bin"
        user_lib_dir.mkdir(parents=True, exist_ok=True)
        _ensure_windows_szip(user_lib_dir)
        
        # Injection for Windows (Python >= 3.8 requires os.add_dll_directory)
        if sys.version_info >= (3, 8):
            try:
                os.add_dll_directory(str(user_lib_dir))
            except AttributeError:
                pass
        else:
            os.environ['PATH'] = f"{user_lib_dir};{os.environ.get('PATH', '')}"

def _ensure_linux_dependencies(target_dir):
    """
    Downloads required Linux libraries via apt-get, extracts them in a tmp folder,
    and places the .so files flatly in target_dir.
    """
    # Define the vector of required packages to satisfy k-Wave's DAG
    required_packages = ['libsz2', 'libaec0', 'libfftw3-single3', 'libfftw3-double3']
    
    # Quick probabilistic check: if we have files matching szip and fftw3, we assume success
    if list(target_dir.glob("libsz.so*")) and list(target_dir.glob("libfftw3*.so*")):
        return

    print("[AOT-biomaps] Missing Linux dependencies detected. Downloading to user-space...")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dir = Path(tmpdirname)
        
        # 1. Download all required packages
        for pkg in required_packages:
            try:
                subprocess.run(["apt-get", "download", pkg], cwd=tmp_dir, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                # If libsz2 fails, try szip as fallback (depends on Ubuntu version)
                if pkg == 'libsz2':
                    try:
                        subprocess.run(["apt-get", "download", "szip"], cwd=tmp_dir, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    except subprocess.CalledProcessError:
                        pass
                continue

        # 2. Extract all downloaded .deb files
        deb_files = list(tmp_dir.glob("*.deb"))
        if not deb_files:
            print("[AOT-biomaps] Warning: Could not download any .deb packages. Acoustic simulations might fail.")
            return
        
        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        
        for deb_file in deb_files:
            try:
                subprocess.run(["dpkg-deb", "-x", str(deb_file), str(extract_dir)], check=True, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                pass
        
        # 3. Find and flatten all .so files into ~/.local/lib
        # This solves the topological problem by projecting the complex tree into a single directory
        installed_files = 0
        for so_file in extract_dir.rglob("*.so*"):
            if so_file.is_file():
                dest = target_dir / so_file.name
                if not dest.exists():
                    # shutil.copy2 resolves symlinks by default, making real copies of the binaries.
                    # This prevents "broken link" issues when the tmp folder is deleted.
                    shutil.copy2(so_file, dest)
                    installed_files += 1
        
        if installed_files > 0:
            print(f"[AOT-biomaps] Successfully installed {installed_files} library files in {target_dir}")

def _ensure_windows_szip(target_dir):
    """
    Downloads and extracts szip.dll for Windows via temporary directory.
    """
    dll_path = target_dir / "szip.dll"
    if dll_path.exists():
        return

    print("[AOT-biomaps] szip.dll not found. Downloading...")
    url = "https://support.hdfgroup.org/ftp/lib-external/szip/2.1.1/bin/windows/szip-2.1.1-win64.zip"
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dir = Path(tmpdirname)
        tmp_zip = tmp_dir / "szip.zip"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(tmp_zip, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
                
            with zipfile.ZipFile(tmp_zip, 'r') as zf:
                zf.extract("szip-2.1.1-win64/bin/szip.dll", path=tmp_dir)
            
            extracted_dll = tmp_dir / "szip-2.1.1-win64" / "bin" / "szip.dll"
            shutil.move(str(extracted_dll), str(dll_path))
            
            print(f"[AOT-biomaps] szip.dll successfully installed at {dll_path}")
        except Exception as e:
            print(f"[AOT-biomaps] Installation error on Windows: {e}")