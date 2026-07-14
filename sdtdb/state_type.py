from dataclasses import dataclass
from typing import Optional


@dataclass
class StateType:
    name: str
    algorithm: str
    flag: Optional[str] = None
    dim_pointer: Optional[int] = None
    pad_factor: Optional[str]=None