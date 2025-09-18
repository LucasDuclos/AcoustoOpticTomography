from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, NVMLError
import psutil

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.numGPUs = 0
            self.bestGPU = None
            self.process = 'cpu'  # Valeur par défaut
            self.numCPUs = psutil.cpu_count(logical=False)
            self.availableMemory = 100 - self.get_memory_usage()
            self.batchSize = self.calculate_batch_size()
            self._init_gpu()

    def _init_gpu(self):
        """Initialise les informations liées au GPU."""
        try:
            nvmlInit()
            self.numGPUs = nvmlDeviceGetCount()
            if self.numGPUs > 0:
                self.process = 'gpu'
                self.bestGPU = self.select_best_gpu()
            else:
                self.process = 'cpu'
                self.bestGPU = None
        except NVMLError as e:
            print(f"NVIDIA GPU not available: {e}")
            self.process = 'cpu'
            self.bestGPU = None
            self.numGPUs = 0
        except Exception as e:
            print(f"Unexpected error during GPU initialization: {e}")
            self.process = 'cpu'
            self.bestGPU = None
            self.numGPUs = 0
        finally:
            try:
                nvmlShutdown()
            except:
                pass  # Évite les erreurs si nvmlShutdown est appelé plusieurs fois

    def set_process(self, process):
        """Définit le processus à utiliser ('cpu' ou 'gpu')."""
        if process not in ['cpu', 'gpu']:
            raise ValueError("process must be 'cpu' or 'gpu'")
        self.process = process

    def get_process(self):
        """Retourne le processus actuel ('cpu' ou 'gpu')."""
        return self.process

    def select_best_gpu(self):
        """Sélectionne le GPU avec le plus de mémoire disponible."""
        try:
            nvmlInit()
            best_gpu = 0
            max_memory = 0
            for i in range(self.numGPUs):
                handle = nvmlDeviceGetHandleByIndex(i)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                available_memory = mem_info.total - mem_info.used
                if available_memory > max_memory:
                    max_memory = available_memory
                    best_gpu = i
            return best_gpu
        except NVMLError as e:
            print(f"Failed to select GPU: {e}")
            return 0  # Retourne le premier GPU par défaut en cas d'erreur
        finally:
            try:
                nvmlShutdown()
            except:
                pass

    def get_memory_usage(self):
        """Retourne l'utilisation actuelle de la mémoire RAM (en pourcentage)."""
        return psutil.virtual_memory().percent

    def calculate_batch_size(self, max_memory_usage=90, min_batch_size=1, max_batch_size=20):
        """Calcule dynamiquement la taille du batch en fonction de la mémoire disponible."""
        if self.availableMemory > max_memory_usage:
            return max_batch_size
        else:
            return max(min_batch_size, int((self.availableMemory / max_memory_usage) * max_batch_size))

# Initialisation unique de la configuration
config = Config()
