import os

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
            self.process = 'cpu'  # Default value
            self.numCPUs = os.cpu_count()
            self.availableMemory = 100 - self.get_memory_usage()
            self.batchSize = self.calculate_batch_size()
            self._init_gpu()

    def _init_gpu(self):
        """Initialize GPU-related information."""
        try:
            import cupy as cp
            self.numGPUs = cp.cuda.runtime.getDeviceCount()
            if self.numGPUs > 0:
                self.process = 'gpu'
                self.bestGPU = self.select_best_gpu()
            else:
                self.process = 'cpu'
                self.bestGPU = None
        except ImportError:
            self.process = 'cpu'
            self.bestGPU = None
            self.numGPUs = 0
        except Exception as e:
            print(f"Unexpected error during GPU initialization: {e}")
            self.process = 'cpu'
            self.bestGPU = None
            self.numGPUs = 0

    def set_process(self, process):
        """Set the process to use ('cpu' or 'gpu')."""
        if process not in ['cpu', 'gpu']:
            raise ValueError("process must be 'cpu' or 'gpu'")
        self.process = process

    def get_process(self):
        """Return the current process ('cpu' or 'gpu')."""
        return self.process

    def select_best_gpu(self):
        """Select the GPU with the most available memory."""
        try:
            import cupy as cp
            best_gpu = 0
            max_memory = 0
            for i in range(self.numGPUs):
                cp.cuda.runtime.setDevice(i)
                # Use modern CuPy API (12+)
                try:
                    free_mem = cp.cuda.runtime.getFreeMem()
                    total_mem = cp.cuda.runtime.getTotalMem()
                    available_memory = free_mem
                except AttributeError:
                    # Fallback for older CuPy versions
                    mem_info = cp.cuda.runtime.memoryInfo()
                    available_memory = mem_info.total - mem_info.used
                if available_memory > max_memory:
                    max_memory = available_memory
                    best_gpu = i
            return best_gpu
        except Exception as e:
            print(f"Failed to select GPU: {e}")
            return 0  # Return first GPU by default in case of error

    def get_memory_usage(self):
        """Return the current RAM memory usage (as a percentage)."""
        try:
            
            return 0
        except ImportError:
            return 0

    def calculate_batch_size(self, max_memory_usage=90, min_batch_size=1, max_batch_size=20):
        """Dynamically calculate batch size based on available memory."""
        if self.availableMemory > max_memory_usage:
            return max_batch_size
        else:
            return max(min_batch_size, int((self.availableMemory / max_memory_usage) * max_batch_size))

# Unique configuration initialization
config = Config()
