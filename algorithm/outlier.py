from typing import TypedDict


class Outlier(TypedDict):
    """
    Data structure representing a detected outlier in the telemetry dataset.
    
    This TypedDict defines the schema for outliers, facilitating type hinting 
    while maintaining the lightweight nature of a dictionary. It includes the 
    timestamp, the actual value, the limits at that time, and the mnemonic 
    identification.

    Attributes:
        time (float): The absolute timestamp of the outlier in seconds.
        t_string (str): Human-readable time tag string.
        value (float): The actual telemetry value at the specified time.
        limit (float): The threshold limit (e.g., error limit) at the time.
        diff (float): The normalized difference between the actual value and the trend.
        mnemonic_id (str): The logical path/ID of the mnemonic.
        time_index (int): The index of the point in the original data stream.
        state (str): The operational state in which the outlier was detected.
    """
    time: float
    t_string: str
    value: float
    limit: float
    diff: float
    mnemonic_id: str
    time_index: int
    state: str
