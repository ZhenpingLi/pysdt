import json
import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import sdt_config


def generate_index(json_db_dir):
    """
    Scans the json_db directory and creates an index of mnemonics per subsystem.
    """
    if not os.path.exists(json_db_dir):
        print(f"Error: Directory not found: {json_db_dir}")
        return

    index_data = {}
    json_files = glob.glob(os.path.join(json_db_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} files in {json_db_dir}...")

    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        # Skip the index file itself if it already exists
        if filename == "mnemonic_index.json":
            continue
            
        subsystem_name = os.path.splitext(filename)[0]
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                mnemonics = []
                if 'mnemonics' in data:
                    for mn in data['mnemonics']:
                        if 'name' in mn:
                            mnemonics.append(mn['name'])
                
                index_data[subsystem_name] = mnemonics
                
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}")
        except Exception as e:
            print(f"Warning: Error processing {filename}: {e}")

    # Write the index file
    output_path = os.path.join(json_db_dir, "mnemonic_index.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(index_data, f, indent=4)
        print(f"Successfully created index at: {output_path}")
        print(f"Indexed {len(index_data)} subsystems.")
    except Exception as e:
        print(f"Error writing index file: {e}")

if __name__ == "__main__":
    # Default path assumes running from project root or util dir
    # Adjust this default to match your likely usage
    default_db_dir = "sdt-config/json_db"
    
    if len(sys.argv) > 1:
        sat_id = sys.argv[1]
        sdt_config.set_sat_id(sat_id)
        target_dir = os.path.join(sdt_config.config_dir, "json_db")
    else:
        target_dir = default_db_dir
        
    generate_index(target_dir)
