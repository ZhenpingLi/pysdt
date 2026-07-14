from dataclasses import dataclass, field
from typing import List

from sdtdb.mnemonic_type import MnemonicType


@dataclass
class SubsystemType:
    """
    Data structure representing the definition of a satellite subsystem.
    
    This dataclass encapsulates the name of the subsystem, the satellite it 
    belongs to, and a list of all telemetry mnemonics associated with it. 
    It is used to load and manage the hierarchical structure of the satellite 
    database.

    Attributes:
        name (str): The name of the subsystem (e.g., 'COM', 'EPS').
        satid (str): The ID of the satellite this subsystem belongs to.
        mnemonics (List[MnemonicType]): A list of MnemonicType objects 
            defining each telemetry point within this subsystem.
    """
    name: str
    satid: str
    mnemonics: List[MnemonicType] = field(default_factory=list)
