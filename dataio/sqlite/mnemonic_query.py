import json
from collections import defaultdict
import os
from typing import Optional, Dict


class MnemonicQuerier:
    """
    A class to efficiently query mnemonic names and get their corresponding IDs.
    It uses an inverted index for fast, O(1) average time complexity lookups.
    """

    def __init__(self, config_path: str):
        """
        Initializes the querier by loading the inverted index file.

        Args:
            index_path (str): The path to the JSON file containing the
                                       inverted mnemonic index.
        """

        self.index_path = os.path.join(config_path, 'json_db', 'mnemonic_index_map.json')
        self._index = None

    @property
    def index(self):
        """Lazy-loads the index from the JSON file."""
        if self._index is None:
            try:
                with open(self.index_path, 'r') as f:
                    self._index = json.load(f)
            except FileNotFoundError:
                print(f"Error: Index file not found at {self.index_path}")
                self._index = {}
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {self.index_path}")
                self._index = {}
        return self._index

    def get_index(self, mnemonic_name) -> Optional[str]:
        """
        Gets the list of integer IDs for a given mnemonic name.

        Args:
            mnemonic_name (str): The mnemonic name to query.

        Returns:
            list: A list of integer IDs associated with the mnemonic name.
                  Returns an empty list if the name is not found.
        """
        if mnemonic_name not in self.index:
            return None
        return self.index[mnemonic_name]

if __name__ == '__main__':
    # Define file paths
    original_index_file = '/mnemonic_index.json'



    # 2. Initialize the querier and perform efficient lookups.
    print("\n--- Querying the efficient index ---")
    querier = MnemonicQuerier()

    # Example queries:
    print(f"IDs for 'ADC_ADPRPRX': {querier.get_ids('ADC_ADPRPRX')}")
    print(f"IDs for 'ADC_ADSTATE': {querier.get_ids('ADC_ADSTATE')}")
    print(f"IDs for 'NON_EXISTENT': {querier.get_ids('NON_EXISTENT')}")