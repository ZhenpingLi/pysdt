import os
import sys
from dataclasses import dataclass

# Add parent directory to path to find sdt_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
DEFAULT = "default"
LONGTERM = 1
TREND = "trend"
ELIMIT = "elimit"
WLIMIT = "wlimit"
WELIMIT = "welimit"
STDDEV = "stddev"
DISJOINT = "disjoint"
INFO = "INFO"
WARNING = "WARNING"
SOCOEF = "socoef"
STAT = "stat"

@dataclass
class AlgorithmData:
    """
    Base dataclass for serializing and storing algorithm-specific training results.
    
    This class acts as a template for holding the minimum information required 
     to identify and reconstruct a trained data trend model. Subclasses 
     extend this to include specific model parameters (weights, coefficients) 
     and session-specific statistics.

    Attributes:
        mnemonic_id (str): The logical path/ID of the telemetry mnemonic.
        alg_name (str): The name of the algorithm used (e.g., 'fbnn', 'ridgefe').
    """
    mnemonic_id : str
    alg_name : str
