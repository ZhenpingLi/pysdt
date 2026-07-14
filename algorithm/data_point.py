import numpy as np

class DataPoint:
    """
    Represents a single data point in a time series.
    
    Attributes:
        time: The timestamp of the data point in seconds (float64).
        data: A NumPy array of float values (float32).
    """
    # __slots__ is the key to reducing memory usage.
    # It prevents the creation of a dynamic __dict__ for every object.
    __slots__ = ['time', 'data']

    def __init__(self, time: float, data: np.ndarray):
        # Python's native float is 64-bit (double precision), which is correct for time.
        self.time = float(time)
        
        # Ensure data is a float32 numpy array for memory efficiency
        if isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                self.data = data.astype(np.float32)
            else:
                self.data = data
        else:
            self.data = np.array(data, dtype=np.float32)

    def __str__(self) -> str:
        return f"Time: {self.time}, Data: {self.data}"
