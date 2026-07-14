from dataclasses import dataclass, field
from typing import List, Optional

from sdtdb.state_type import StateType


@dataclass
class AlgorithmType:
    """
    Data structure representing the static definition of a training algorithm.
    
    This dataclass encapsulates the configuration parameters for a specific 
    algorithm as defined in the satellite database. It includes its name, 
    dimensionality, associated operational states, and various attributes 
    that control its behavior.

    Attributes:
        name (str): The name of the algorithm (e.g., 'fbnn', 'ridgefe', 'default').
        dim (List[int]): A list of integers defining the algorithm's dimensions 
            (e.g., hidden layer sizes for neural networks, Fourier expansion order).
        state (List[StateType]): A list of StateType objects defining different 
            operational states the algorithm can model.
        attribute (List[str]): A list of key-value pair strings (e.g., 'key|value') 
            for additional algorithm-specific configurations.
        option (Optional[str]): A pipe-separated string of flags/options 
            (e.g., 'NORMALIZE|RETRAIN').
        stat (Optional[str]): A string indicating a specific statistic to use.
        np (Optional[int]): The number of pattern periods to consider in training.
    """
    name: str
    dim: List[int] = field(default_factory=list)
    state: List[StateType] = field(default_factory=list)
    attribute: List[str] = field(default_factory=list) # List of "name|value" strings
    option: Optional[str] = None
    stat: Optional[str] = None
    np: Optional[int] = None
