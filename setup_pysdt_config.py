import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict

# Configuration for the setup script
PYSDT_HOME = Path.home() / ".pysdt"
APP_JSON_PATH = PYSDT_HOME / "application.json"
DEFAULT_PROJECT_CONFIG_DIR_NAME = "sdt-config"

def setup_pysdt_config():
    """
    Guides the user through setting up a specific satellite configuration.
    It creates a satellite-specific directory, copies default configs,
    and maps the satellite ID to that directory in ~/.pysdt/application.json.
    """
    print("--- PySDT Configuration Setup ---")
    print("This utility sets up the configuration for a specific satellite.")
    print("-" * 30)

    # --- Step 1: Get the Satellite ID ---
    satellite_id: Optional[str] = None
    while not satellite_id:
        user_input = input("Enter the Satellite ID (e.g., G16, NOAA20): ").strip()
        if not user_input:
            print("Satellite ID cannot be empty.")
            continue
        satellite_id = user_input.upper()

    # --- Step 2: Get and validate the Target Configuration Directory ---
    dest_satellite_path: Optional[Path] = None
    while dest_satellite_path is None:
        suggested_path = Path.cwd() / "pysdt_configs"
        user_input = input(f"Enter the root path where the configuration for '{satellite_id}' should be created (e.g., {suggested_path}): ").strip()

        if not user_input:
            print("A target directory path is required.")
            continue

        # Resolve to absolute path and include the satellite ID as a subdirectory
        config_base_path = Path(user_input).resolve()
        dest_satellite_path = config_base_path / satellite_id.lower()

        # Create the directory
        try:
            dest_satellite_path.mkdir(parents=True, exist_ok=True)
            print(f"Created/Ensured satellite directory: {dest_satellite_path}")
        except OSError as e:
            print(f"Error creating directory '{dest_satellite_path}': {e}. Please check permissions.")
            dest_satellite_path = None
            continue

    # --- Step 3: Copy default content to the new satellite subdirectory ---
    source_config_dir = Path(__file__).resolve().parent / DEFAULT_PROJECT_CONFIG_DIR_NAME
    
    if source_config_dir.is_dir():
        print(f"Copying default configuration from '{source_config_dir}' to '{dest_satellite_path}'...")
        try:
            shutil.copytree(source_config_dir, dest_satellite_path, dirs_exist_ok=True)
            print("Default configuration files copied successfully.")
        except Exception as e:
            print(f"Error copying configuration files: {e}")
    else:
        print(f"Warning: Project's default '{DEFAULT_PROJECT_CONFIG_DIR_NAME}' directory not found at '{source_config_dir}'. Skipping copy.")

    # --- Step 4: Update/Create application.json in the home directory ---
    PYSDT_HOME.mkdir(parents=True, exist_ok=True)
    app_mapping: Dict[str, str] = {}

    # Check if application.json exists
    if APP_JSON_PATH.is_file():
        print(f"Existing configuration registry found at {APP_JSON_PATH}. Appending...")
        try:
            with open(APP_JSON_PATH, 'r') as f:
                app_mapping = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read existing mapping: {e}. Starting fresh.")
            app_mapping = {}
    else:
        print(f"Creating new configuration registry at {APP_JSON_PATH}...")

    # Map the satellite ID to the ABSOLUTE path of its configuration directory
    app_mapping[satellite_id.lower()] = str(dest_satellite_path)
    
    try:
        with open(APP_JSON_PATH, 'w') as f:
            json.dump(app_mapping, f, indent=4)
        print(f"Successfully updated registry with: {satellite_id.lower()} -> {dest_satellite_path}")
    except IOError as e:
        print(f"Error writing to '{APP_JSON_PATH}': {e}")

    print("\n--- Setup Complete ---")
    print(f"To use this configuration, run PySDT with: sdt {satellite_id} [command]")

if __name__ == "__main__":
    setup_pysdt_config()
