from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
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
            try:
                nvmlInit()
                self.numGPUs = nvmlDeviceGetCount()
                if self.numGPUs == 0:
                    self.process = 'cpu'
                    self.bestGPU = None
                else:
                    self.process = 'gpu'
                    self.bestGPU = self.select_best_gpu()
                nvmlShutdown()
            except Exception as e:
                print("GPU init failed:", e)
                self.process = 'cpu'
                self.bestGPU = None
                self.numGPUs = 0

            self.numCPUs = psutil.cpu_count(logical=False)
            self.availableMemory = 100 - self.get_memory_usage()
            self.batchSize = self.calculate_batch_size()

    def set_process(self, process):
        if process not in ['cpu', 'gpu']:
            raise ValueError("process must be 'cpu' or 'gpu'")
        self.process = process

    def get_process(self):
        return self.process

    def select_best_gpu(self):
        nvmlInit()
        best_gpu = 0
        max_memory = 0
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            available_memory = mem_info.total - mem_info.used
            if available_memory > max_memory:
                max_memory = available_memory
                best_gpu = i
        nvmlShutdown()
        return best_gpu

    def get_memory_usage(self):
        """Returns the current memory usage in percentage (RAM, not GPU)."""
        return psutil.virtual_memory().percent

    def calculate_batch_size(self, max_memory_usage=90, min_batch_size=1, max_batch_size=20):
        """Calculate dynamic batch size based on available memory."""
        if self.availableMemory > max_memory_usage:
            return max_batch_size
        else:
            return max(min_batch_size, int((self.availableMemory / max_memory_usage) * max_batch_size))


config = Config()
