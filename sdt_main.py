"""
SDT: Mission Telemetry Modeling and Analysis Tool.

This is the primary entry point for the AIMS-SDT application. It provides both 
a command-line (batch) interface and an interactive shell for performing 
data training, generating visualizations, and managing model archives for 
satellite missions.

Key features:
*   Parallel data training using Ray.
*   Interactive shell with command history and tab completion.
*   Pluggable architecture for training algorithms and data storage.
*   Advanced telemetry visualization (trends, outliers, events).
"""

import argparse
import atexit
import gc
import logging
import os
import sys
from pathlib import Path
from typing import List, Any

from sdt_exception import SDTException
from util.time_util import get_time_from_string

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config.sdt_config as sdt_config
from sdtdb import sdt_db
import training.data_buffer as data_buffer
from training.sdt_training_session import SDTTrainingSession
from dataio.data_training_io import DataTrainingIO

def add_override_message_to_hook():
    """Adds a status message to the Python interactive hook."""
    try:
        old_hook = sys.__interactivehook__
    except AttributeError:
        return
    def hook():
        old_hook()
        print("Using GNU readline instead of the default readline (see usercustomize.py)")
    sys.__interactivehook__ = hook

# --- Interactive Mode Enhancements ---
try:
    import gnureadline as readline
    add_override_message_to_hook()
except ImportError:
    import readline
sys.modules["readline"] = readline

# --- Constants ---
CONTEXT = "MTMAMain"
VERSION = "1.0.0"
HISTORY_FILE = os.path.join(os.path.expanduser("~"), ".sdt_history")


# --- Global Resource Tracker ---
active_resources: List[Any] = []
"""List of objects that require explicit closing at application exit."""

def register_resource(resource: Any):
    """
    Registers a resource to be closed during application cleanup.

    Args:
        resource (Any): An object with a 'close' method.
    """
    if resource and hasattr(resource, 'close'):
        active_resources.append(resource)

def setup_logging(level=logging.INFO):
    """
    Configures the root logger for the application.

    Args:
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def handle_training_request(args: argparse.Namespace):
    """
    Orchestrates a training session based on provided command arguments.
    Handles both 'train' (short-term) and 'lttrain' (long-term) commands.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logging.info(f"Initiating '{args.command}' for satellite '{sdt_config.sat_id}'...")

    session_type = data_buffer.LONGTERM if args.command == 'lttrain' else data_buffer.SHORTTERM
    data_buffer.set_session_type(session_type)

    session = SDTTrainingSession()
    try:
        session.perform_training(args)
    except Exception as e:
        logging.error(f"{CONTEXT}: A critical error occurred during the session: {e}", exc_info=True)

def handle_plot_request(args: argparse.Namespace):
    """
    Dispatches plotting requests to the appropriate handlers.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing plot_type and IDs.
    """
    logging.info(f"Received plot command: '{args.plot_type}' for satellite '{sdt_config.sat_id}'")
    
    from dataplot.data_plot import HandleDataPlot
    from dataplot.handle_outlier_plot import HandleOutlierPlot
    
    if args.plot_type == "data":
        if args.ids:
            plot_handler = HandleDataPlot(args.ids[0])
            plot_handler.run()
    elif args.plot_type == "outlier":
        id_list = args.ids if args.ids else sdt_db.get_dataset_name_list()
        outlier_plot = HandleOutlierPlot(id_list)
        outlier_plot.run()

def handle_save_request(args: argparse.Namespace):
    """
    Handles the 'save' command to persist trained models from memory to the archive.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing IDs to save.
    """
    logging.info(f"Received save command for satellite '{sdt_config.sat_id}' with IDs: {args.ids}")

    try:
        training_io = DataTrainingIO()
        register_resource(training_io)

        with training_io:
            for id_str in args.ids:
                if sdt_db.get_mnemonic_type(id_str):
                    training_output_data = data_buffer.get_training_output_data(mnemonic_id=id_str)
                    if training_output_data:
                        logging.info(f"Saving data for {id_str}...")
                        training_io.save([training_output_data])
                    else:
                        logging.warning(f"Data for '{id_str}' not found in buffer. Was it trained?")
                else:
                    logging.warning(f"ID '{id_str}' is not defined in the AIMS database.")
                    
    except SDTException as e:
        logging.error(f"A data I/O error occurred during save: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during save operation: {e}", exc_info=True)

def handle_debug_request(args: argparse.Namespace):
    """
    Toggles the application logging verbosity between INFO and DEBUG.

    Args:
        args (argparse.Namespace): Arguments containing the target 'state' ('on' or 'off').
    """
    if args.state.lower() == 'on':
        setup_logging(logging.DEBUG)
        sdt_config.debug = "on"
        logging.info("Debug mode enabled.")
    elif args.state.lower() == 'off':
        setup_logging(logging.INFO)
        sdt_config.debug = "off"
        logging.info("Debug mode disabled.")
    else:
        print("Usage: debug [on|off]")

def create_parser() -> argparse.ArgumentParser:
    """
    Initializes and configures the ArgumentParser for all tool commands.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=f"SDT v{VERSION} - Data Training and Monitoring Tool",
        epilog="Use '<command> --help' for more information on a specific command."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Training Parser
    train_parser = subparsers.add_parser("train", help="Short-term training.")
    train_parser.add_argument("ids", nargs='*', help="IDs to train (omit for all).")
    train_parser.add_argument("-s", "--start", type=get_time_from_string, help="Start time (YYYY/DDD).")
    train_parser.add_argument("-e", "--end", type=get_time_from_string, help="End time (YYYY/DDD).")
    train_parser.add_argument("--input_time", type=get_time_from_string, help="Input model timestamp.")
    train_parser.add_argument("--inputid", help="Input model ID.")
    train_parser.add_argument("--mode", choices=['auto', 'manual'], default='auto')
    train_parser.set_defaults(func=handle_training_request)

    # LT Training Parser
    lttrain_parser = subparsers.add_parser("lttrain", help="Long-term training.")
    lttrain_parser.add_argument("ids", nargs='*', help="IDs to train.")
    lttrain_parser.set_defaults(func=handle_training_request)

    # Plotting Parser
    plot_parser = subparsers.add_parser("plot", help="Generate visualizations.")
    plot_parser.add_argument("plot_type", choices=['data', 'outlier', 'event'], help="Type of plot.")
    plot_parser.add_argument("ids", nargs='*', help="IDs to plot.")
    plot_parser.set_defaults(func=handle_plot_request)
    
    # Saving Parser
    save_parser = subparsers.add_parser("save", help="Archive trained models.")
    save_parser.add_argument("ids", nargs='+', help="IDs to save.")
    save_parser.set_defaults(func=handle_save_request)

    # Debugging Parser
    debug_parser = subparsers.add_parser("debug", help="Toggle debug logging.")
    debug_parser.add_argument("state", choices=['on', 'off'])
    debug_parser.set_defaults(func=handle_debug_request)

    return parser

def setup_readline():
    """Initializes command history support for the interactive shell."""
    if 'readline' in sys.modules:
        try:
            readline.read_history_file(HISTORY_FILE)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
        atexit.register(readline.write_history_file, HISTORY_FILE)

def cleanup():
    """
    Performs application-wide resource cleanup before exit.
    Closes database connections and triggers garbage collection.
    """
    logging.info("Cleaning up resources...")
    for resource in active_resources:
        try:
            resource.close()
        except Exception as e:
            logging.warning(f"Error closing resource: {e}")
    gc.collect()

def run_interactive_session(parser: argparse.ArgumentParser, sat_id: str):
    """
    Starts an interactive REPL session for the specified satellite.

    Args:
        parser (argparse.ArgumentParser): The command parser.
        sat_id (str): The active satellite ID.
    """
    setup_readline()
    atexit.register(cleanup)
    
    print(f"--- MTMA v{VERSION} Interactive Mode ---")
    print(f"Satellite context: {sat_id}")
    print("Enter 'quit' or 'exit' to leave.")
    
    while True:
        try:
            cmd_line = input(f"sdt({sat_id})> ")
            if cmd_line.lower() in ['quit', 'exit']:
                break
            if not cmd_line.strip():
                continue
            
            if cmd_line.strip() in ['help', '--help', '-h']:
                print("\nAvailable commands:")
                print("  train    - Perform short-term data training")
                print("  lttrain  - Perform long-term data training")
                print("  plot     - Generate data, outlier, or event plots")
                print("  save     - Save trained models to the archive")
                print("  debug    - Toggle verbose debug logging (on/off)")
                print("  exit     - Exit the session\n")
                continue

            tokens = cmd_line.split()
            args = parser.parse_args(tokens)
            
            if hasattr(args, 'func'):
                args.func(args)
            else:
                print(f"Unknown command '{tokens[0]}'. Type 'help' for a list of commands.")

        except SystemExit:
            pass # Prevent --help from closing the session
        except Exception as e:
            logging.error(f"An error occurred in interactive mode: {e}", exc_info=True)
    
    cleanup()

def main():
    """
    Main execution entry point. Handles satellite context initialization and 
    dispatches between interactive and batch modes.
    """
    if len(sys.argv) < 2:
        print("Usage: sdt_main.py <satellite_id> [command] [options]")
        print("Error: Satellite ID is a required first argument.")
        sys.exit(1)

    # Handle global --help directly to list commands
    if sys.argv[1] in ['--help', '-h']:
        print(f"SDT v{VERSION} - Available Commands:")
        print("  train    - Perform short-term data training")
        print("  lttrain  - Perform long-term data training")
        print("  plot     - Generate various visualizations")
        print("  save     - Save models to database")
        print("  debug    - Toggle verbose logging")
        print("\nUsage for specific command: python sdt_main.py <sat_id> <command> --help")
        sys.exit(0)

    sat_id = sys.argv[1]
    sdt_config.set_sat_id(sat_id)

    parser = create_parser()

    if len(sys.argv) == 2:
        setup_logging()
        logging.info(f"--- SDT v{VERSION} Initializing for satellite '{sat_id}' ---")
        run_interactive_session(parser, sat_id)
    else:
        args = parser.parse_args(sys.argv[2:])
        log_level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(log_level)
        logging.info(f"--- AIMS-SDT v{VERSION} Initializing for satellite '{sat_id}' (Batch Mode) ---")
        
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()
    
    cleanup()

if __name__ == "__main__":
    main()
