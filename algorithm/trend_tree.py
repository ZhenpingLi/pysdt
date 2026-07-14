import os
import sys
from typing import Optional

from algorithm.mnemonic_node import MnemonicNode

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdtdb import sdt_db
from algorithm.trend_node import TrendNode

# Constants
LONGTERM = 1
MNPOSTFIX = ["max", "min", "mean", "sigma"]

class TrendTree:
    """
    Utility class to build and manage hierarchical training trees.
    
    The tree structure reflects the physical organization of the satellite, 
    mapping subsystems to their individual telemetry mnemonics. This hierarchy 
    is used to orchestrate data training and analysis.
    """
    CONTEXT = "DTTREE"

    @classmethod
    def build_tree(cls, id_string: str) -> Optional[TrendNode]:
        """
        Entry point to build a training tree for a given identifier.
        
        It determines if the ID corresponds to a whole subsystem or a single 
        mnemonic and invokes the appropriate builder method.

        Args:
            id_string (str): The logical identifier (e.g., 'COM' or 'COM/TEMP_1').

        Returns:
            Optional[TrendNode]: The root of the constructed tree, or None if the ID is invalid.
        """
        if sdt_db.is_subsystem(id_string):
            return cls._build_subsystem_tree(id_string)
        elif sdt_db.get_mnemonic_type(id_string) is not None:
            return cls._build_mnemonic_tree(id_string)
        else:
            return None

    @classmethod
    def _build_subsystem_tree(cls, sub_system_name: str) -> TrendNode:
        """
        Builds a hierarchical tree representing an entire satellite subsystem.
        
        Retrieves the subsystem definition from the database and creates a root 
        TrendNode with child MnemonicNodes for every mnemonic in that subsystem.

        Args:
            sub_system_name (str): The name of the subsystem (e.g., 'POWER').

        Returns:
            TrendNode: A node representing the subsystem with its children populated.
        """
        data_set = sdt_db.get_dataset(sub_system_name)
        sub_system_node = TrendNode(sub_system_name)
        if data_set:
            mn_node_list = []
            for mn_type in data_set.mnemonics:
                mn_name = mn_type.name
                mn_node = cls._build_mnemonic_tree(mn_name)
                mn_node_list.append(mn_node)
            sub_system_node.children = mn_node_list
        return sub_system_node

    @classmethod
    def _build_mnemonic_tree(cls, mnemonic_name: str) -> MnemonicNode:
        """
        Builds a leaf node (or sub-root) for a specific telemetry mnemonic.

        Args:
            mnemonic_name (str): The mnemonic identifier.

        Returns:
            MnemonicNode: The constructed mnemonic node.
        """
        mn_node = MnemonicNode(mnemonic_name)
        return mn_node
