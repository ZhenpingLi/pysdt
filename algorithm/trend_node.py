from __future__ import annotations

from typing import List


class TrendNode:
    """
    Represents a node in the hierarchical training tree.
    
    A TrendNode can represent either a satellite subsystem (container of mnemonics) 
    or a specific telemetry mnemonic. It facilitates the recursive processing 
    of data training and analysis.
    """

    SUBSYSTEM = 1
    MNEMONIC = 2

    def __init__(self, _name: str):
        """
        Initializes a new instance of the TrendNode.

        Args:
            _name (str): The name of the node (e.g., 'COM' or 'TEMP_1').
        """
        self.name = _name
        self.children : List[TrendNode] = []

    def add_child(self, child: TrendNode):
        """
        Adds a child node to the current node's hierarchy.

        Args:
            child (TrendNode): The node to be added as a child.
        """
        self.children.append(child)

    def get_node_level(self) -> int:
        """
        Determines the structural level of the current node based on its children.

        Returns:
            int: SUBSYSTEM if the node has children, otherwise MNEMONIC.
        """
        return TrendNode.SUBSYSTEM if len(self.children) > 0 else TrendNode.MNEMONIC

    @property
    def is_leaf(self) -> bool:
        """
        Checks if the node is a leaf (has no children).

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return len(self.children) == 0

    def get_children(self) -> List[TrendNode]:
        """Returns the list of child nodes."""
        return self.children
