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
        
    def __repr__(self):
        """Display the configuration."""
        lines = [
            "=" * 60,
            "AOT_biomaps Configuration",
            "=" * 60,
            f"Process:        {self.process.upper()}",
            "",
            "Hardware:",
            f"  CPUs:         {self.numCPUs}",
            f"  GPUs:         {self.numGPUs}",
        ]

        # Add GPU details if available
        if self.numGPUs > 0 and self.bestGPU is not None:
            try:
                gpu_info = self.get_gpu_info(self.bestGPU)
                if gpu_info:
                    lines.extend([
                        f"  Best GPU:     {self.bestGPU}",
                        "",
                        "GPU Details:",
                        f"  Name:         {gpu_info.get('name', 'N/A')}",
                        f"  Memory:       {gpu_info.get('free_memory_mb', 0)} MB free / {gpu_info.get('total_memory_mb', 0)} MB total",
                        f"  Compute:      {gpu_info.get('compute_capability', 'N/A')}",
                    ])
            except:
                pass

        lines.extend([
            "",
            "Memory:",
            f"  Available:    {self.availableMemory:.1f}%",
            "",
            "Performance:",
            f"  Batch Size:  {self.batchSize}",
            "=" * 60,
        ])
        return "\n".join(lines)

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
                self.numGPUs = 0
        except ImportError:
            self.process = 'cpu'
            self.bestGPU = None
            self.numGPUs = 0
        except Exception:
            # Silently fall back to CPU
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
                # Use modern CuPy API
                try:
                    free_mem = cp.cuda.runtime.getFreeMem()
                    total_mem = cp.cuda.runtime.getTotalMem()
                    available_memory = free_mem
                except AttributeError:
                    # Fallback for older CuPy versions
                    try:
                        free_mem, total_mem = cp.cuda.runtime.memoryGetInfo()
                        available_memory = free_mem
                    except AttributeError:
                        # Very old CuPy - use memoryInfo as last resort
                        mem_info = cp.cuda.runtime.memoryInfo()
                        available_memory = mem_info.total - mem_info.used
                if available_memory > max_memory:
                    max_memory = available_memory
                    best_gpu = i
            return best_gpu
        except Exception:
            # Silently return first GPU
            return 0  # Return first GPU by default in case of error

    def select_gpu(self, device_id=None):
        """
        Select a specific GPU by ID, or automatically select the best one.
        
        Args:
            device_id (int, optional): GPU device ID to select. If None, selects the best GPU.
            
        Returns:
            int: The selected GPU device ID, or None if no GPU available.
            
        Raises:
            ValueError: If device_id is invalid or out of range.
            RuntimeError: If no GPUs are available.
            
        Examples:
            >>> config.select_gpu()  # Auto-select best GPU
            >>> config.select_gpu(0)  # Select first GPU
            >>> config.select_gpu(1)  # Select second GPU
        """
        if self.numGPUs == 0:
            raise RuntimeError("No GPUs available. Check if CuPy is installed and CUDA is working.")
        
        if device_id is None:
            # Auto-select best GPU
            device_id = self.select_best_gpu()
        else:
            # Validate device_id
            if not isinstance(device_id, int):
                raise ValueError(f"device_id must be an integer, got {type(device_id)}")
            if device_id < 0 or device_id >= self.numGPUs:
                raise ValueError(
                    f"device_id {device_id} is out of range. "
                    f"Available GPUs: 0 to {self.numGPUs - 1}"
                )
        
        # Set the selected GPU
        try:
            import cupy as cp
            cp.cuda.runtime.setDevice(device_id)
            self.bestGPU = device_id
            self.process = 'gpu'
            return device_id
        except Exception as e:
            raise RuntimeError(f"Failed to set GPU {device_id}: {e}")

    def get_gpu_info(self, device_id=None):
        """
        Get information about a specific GPU or the current GPU.
        
        Args:
            device_id (int, optional): GPU device ID. If None, uses current GPU.
            
        Returns:
            dict: Dictionary containing GPU information (name, total_memory, free_memory, etc.)
                   Returns None if no GPU available.
        """
        if self.numGPUs == 0:
            return None
        
        try:
            import cupy as cp
            if device_id is None:
                device_id = self.bestGPU if self.bestGPU is not None else 0
            
            # Save current device
            current_device = cp.cuda.runtime.getDevice()
            
            # Set target device
            cp.cuda.runtime.setDevice(device_id)
            
            # Get device properties
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            
            # Get memory info
            try:
                free_mem = cp.cuda.runtime.getFreeMem()
                total_mem = cp.cuda.runtime.getTotalMem()
            except Exception:
                try:
                    free_mem, total_mem = cp.cuda.runtime.memoryGetInfo()
                except Exception:
                    free_mem = 0
                    total_mem = 0
            
            # Restore current device
            cp.cuda.runtime.setDevice(current_device)
            
            return {
                'device_id': device_id,
                'name': props.name.decode('utf-8'),
                'total_memory_mb': total_mem // (1024 * 1024),
                'free_memory_mb': free_mem // (1024 * 1024),
                'used_memory_mb': (total_mem - free_mem) // (1024 * 1024),
                'total_memory_gb': total_mem / (1024 ** 3),
                'free_memory_gb': free_mem / (1024 ** 3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multiProcessorCount,
                'max_threads_per_block': props.maxThreadsPerBlock,
                'max_block_dim': (props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]),
                'max_grid_dim': (props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]),
            }
        except Exception:
            return None

    def list_gpus(self):
        """
        List all available GPUs with their information.
        
        Returns:
            list: List of dictionaries, each containing GPU information.
                  Returns empty list if no GPUs available.
        """
        if self.numGPUs == 0:
            return []
        
        gpus = []
        for i in range(self.numGPUs):
            info = self.get_gpu_info(i)
            if info:
                gpus.append(info)
        return gpus

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
