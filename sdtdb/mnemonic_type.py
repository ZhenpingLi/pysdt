from dataclasses import dataclass, field
from typing import List, Optional

from sdtdb.algorithm_type import AlgorithmType


@dataclass
class MnemonicType:
    """
    Data structure representing the static definition of a telemetry mnemonic.
    
    This dataclass holds the configuration parameters for a specific mnemonic 
    as defined in the satellite database. It includes its name, sampling 
    frequency, statistical limits, associated training algorithms, and 
    relationships with other telemetry points.

    Attributes:
        name (str): The unique identifier/name of the mnemonic.
        frequency (float): The expected sampling frequency in seconds.
        warning_limit (float): The default warning multiplier for statistical limits.
        error_limit (float): The default error multiplier for statistical limits.
        sml (AlgorithmType): The primary training algorithm for the mnemonic.
        depends (str): The name of a dependent mnemonic or data source.
        attributes (List[str]): A list of key-value pair strings (e.g., 'key|value') 
            defining specific algorithm behaviors.
        lml (Optional[List[AlgorithmType]]): A list of secondary algorithms 
            associated with the mnemonic (e.g., for long-term trending).
    """
    name: str
    frequency: float
    warning_limit: float
    error_limit: float
    sml: AlgorithmType
    depends: str
    attributes: List[str] = field(default_factory=list) # List of "name|value" strings
    lml: Optional[List[AlgorithmType]] = field(default_factory=list)
