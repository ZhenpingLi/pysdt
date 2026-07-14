from dataclasses import dataclass
import numpy as np


@dataclass
class MnemonicStatus:
    """
    Data structure representing the operational status of a telemetry mnemonic.
    
    This dataclass encapsulates the health metrics calculated for a mnemonic 
    at a specific point in time, typically at the end of a training session. 
    It includes the aggregate status across various dimensions (e.g., TPC, 
    outlier clusters).

    Attributes:
        time (float): The absolute timestamp for this status report in Unix seconds.
        mnemonic_id (str): The logical path or identifier of the mnemonic.
        state_id (str): The operational state associated with this status (e.g., 'default').
        status_array (np.ndarray): A NumPy array containing numeric status metrics 
            (e.g., [ratio, outlier_value, status_level]).
    """
    time: float
    mnemonic_id: str
    state_id: str
    status_array: np.ndarray
