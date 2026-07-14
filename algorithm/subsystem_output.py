from dataclasses import dataclass, field
from typing import List

from algorithm.training_output import TrainingOutputData


@dataclass
class SubsystemOutput:
    """
    Data structure representing the aggregated training results for an entire satellite subsystem.
    
    This object acts as a container for the individual training outputs of all 
    mnemonics belonging to a specific subsystem (e.g., 'COM', 'POWER').
    
    Attributes:
        subsystem_name (str): The name of the subsystem.
        mnemonic_output_list (List[TrainingOutputData]): A list of training results 
            for each mnemonic in the subsystem.
    """
    subsystem_name : str
    mnemonic_output_list: List[TrainingOutputData] = field(default_factory=list)
