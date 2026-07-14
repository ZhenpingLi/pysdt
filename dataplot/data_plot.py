import logging
import multiprocessing
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

import plugin_manager
from sdtdb import sdt_db
from training.preprocessing import data_model_utility

# NOTE: This implementation requires the matplotlib library.
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("ERROR: matplotlib is not installed. Please run 'pip install matplotlib'")
    plt = None

from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
import training.data_buffer as data_buffer

# --- Constants ---
CONTEXT = "DataPlot"

def _worker_plot_process(title: str, xlabel: str, ylabel: str,
                         raw_times: List[datetime], raw_values: List[float],
                         trend_times: List[datetime], trend_values: List[float],
                         upper_err: Optional[List[float]], lower_err: Optional[List[float]],
                         upper_warn: Optional[List[float]], lower_warn: Optional[List[float]]):
    """
    Worker function executed in a separate process to render the Matplotlib window.
    
    This ensures that the interactive GUI loop runs on its own main thread, 
    preventing it from blocking the AIMS-SDT main process or interactive shell.

    Args:
        title (str): The plot title.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        raw_times (List[datetime]): Timestamps for raw telemetry points.
        raw_values (List[float]): Raw telemetry values.
        trend_times (List[datetime]): Timestamps for the trained trend line.
        trend_values (List[float]): Predicted trend values.
        upper_err (Optional[List[float]]): Error limit upper boundary.
        lower_err (Optional[List[float]]): Error limit lower boundary.
        upper_warn (Optional[List[float]]): Warning limit upper boundary.
        lower_warn (Optional[List[float]]): Warning limit lower boundary.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return

    # Create a wide, compact figure
    fig, ax = plt.subplots(figsize=(15, 5))

    # --- Interactive Annotation Setup ---
    annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(x, y):
        """Updates the hover annotation text with precise time and value."""
        dt = mdates.num2date(x, tz=timezone.utc)
        annot.set_text(f"Time: {dt.strftime('%Y/%j %H:%M:%S')}\nValue: {y:.4f}")
        annot.xy = (x, y)
        annot.set_visible(True)
        fig.canvas.draw_idle()

    def on_motion(event):
        """Event handler for mouse motion to trigger annotations."""
        if event.inaxes == ax:
            # Check if mouse is over raw data points (index 0)
            cont, ind = ax.get_children()[0].contains(event)
            if cont:
                x, y = ax.get_children()[0].get_data()
                point_index = ind["ind"][0]
                update_annot(x[point_index], y[point_index])
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    # Connect interactivity
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    # 1. Plot raw telemetry data
    if raw_times and raw_values:
        ax.plot(raw_times, raw_values, 'o', color='red', markersize=2, alpha=0.6, label='Raw Data')

    # 2. Plot the trained trend model
    if trend_times and trend_values:
        ax.plot(trend_times, trend_values, color='blue', linewidth=2, label='Trained Trend')

    # 3. Apply formatting and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    # X-axis: Format as Day-of-Year/Hour
    ax.tick_params(axis='x', rotation=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%j/%H', tz=timezone.utc))

    plt.tight_layout()
    plt.show()


def get_trend_index(trend_times: List[float], time: float):
    if len(trend_times)==1:
        return 0
    else:
        return np.searchsorted(trend_times, time, side='right') - 1


class HandleDataPlot:
    """
    Handler for telemetry and trend visualization requests.
    
    This class orchestrates the retrieval of data from the archive and 
    spawns separate processes to display interactive Matplotlib windows.
    """

    def __init__(self, tokens: List[str]):
        """
        Initializes the plot handler.

        Args:
            tokens (List[str]): List of arguments (e.g., ["mnemonic"]).
        """
        if len(tokens) >= 1:
            self.mnemonic_id = "".join(tokens)
        else:
            self.mnemonic_id = None
        logging.info(f"The mnemonic id: {self.mnemonic_id} and tokens: {tokens}")

    def run(self):
        """
        Main execution method. Retrieves data and trends for the mnemonic 
        and launches the plotting process.
        """
        if not self.mnemonic_id or not sdt_db.exist(self.mnemonic_id):
            logging.error(f"{CONTEXT}: Invalid or missing mnemonic ID for plotting.")
            return
            
        data_input = plugin_manager.get_sdt_data_input("default")
        if data_input is None:
            return
            
        # Retrieve raw telemetry for the session
        training_set = data_input.get_data(self.mnemonic_id, data_buffer.session_start, data_buffer.session_end)
        data_input.close()
        
        # Check if we have a trained trend in the buffer
        training_output = data_buffer.get_training_output_data(self.mnemonic_id)
        if training_output is not None:
            trend_lists = [plugin_manager.get_data_trend_from_output(alg_data) for alg_data in training_output.algorithm_data_list]
            if trend_lists:
                logging.info(f"Preparing trend plot for {self.mnemonic_id}...")
                self.plot_data_trend(training_set, trend_lists)
            else:
                logging.warning(f"No trained trend found; plotting raw data only.")
                self.plot_data(training_set, self.mnemonic_id)
        else:
            self.plot_data(training_set, self.mnemonic_id)

    @staticmethod
    def _to_seconds(ts: float) -> float:
        """Heuristic to normalize millisecond timestamps to seconds."""
        if ts > 100_000_000_000:
            return ts / 1000.0
        return ts

    @staticmethod
    def plot_training_set(training_set: TrainingSet, title: str = "Training Set"):
        """
        Plots a standalone TrainingSet in a new process.

        Args:
            training_set (TrainingSet): The dataset to visualize.
            title (str): Optional custom window title.
        """
        if not training_set or training_set.inputs.size == 0:
            return

        # Prepare data for transfer to the child process
        raw_times = [datetime.fromtimestamp(HandleDataPlot._to_seconds(t[0]), tz=timezone.utc) for t in training_set.inputs]
        raw_values = training_set.outputs.tolist()

        p = multiprocessing.Process(
            target=_worker_plot_process,
            args=(title, "Time (UTC Year/DOY HH:MM)", "Value",
                  raw_times, raw_values,
                  [], [], None, None, None, None)
        )
        p.start()

    @staticmethod
    def plot_data_trend(training_set: TrainingSet, trends: List[DataTrend]):
        """
        Prepares and displays a combined plot of raw telemetry and its trained trend.

        Args:
            training_set (TrainingSet): Raw data.
            trends (List[DataTrend]): List of trained model components.
        """
        if not trends:
             return
             
        # Ensure input features are updated for trend calculation
        data_model_utility.get_model_inputs(training_set, algorithm=trends[0].algorithm, mnemonic_id=trends[0].mnemonic_id)
        
        trend_times = [trend.get_reference_time() for trend in trends]
        lg_path_str = trends[0].mnemonic_id
        raw_times = training_set.inputs
        raw_values = training_set.raw
        
        # 1. Prepare raw data timestamps
        raw_times_dt = [datetime.fromtimestamp(t[0], tz=timezone.utc) for t in raw_times]
        
        # 2. Sample 500 points for a smooth trend line
        indices = np.linspace(0, len(raw_times) - 1, num=500, dtype=int)
        trend_input_points = raw_times[indices]
        trend_times_dt = [datetime.fromtimestamp(t[0], tz=timezone.utc) for t in trend_input_points]

        # 3. Calculate trend and statistical boundaries
        trend_values : List[float] =[]
        upper_warn : List[float] =[]
        lower_warn : List[float] =[]
        for time in trend_input_points:
            idx = get_trend_index(trend_times, time[0])
            trend = trends[idx]
            values = trend.get_trend_values_at(time, "wlimit")
            if values is not None:
                trend_values.append(values[1])
                upper_warn.append(values[0])
                lower_warn.append(values[2])

        # 4. Spawn the interactive window process
        logging.info(f"{CONTEXT}: Launching plot window for {lg_path_str}")
        p = multiprocessing.Process(
            target=_worker_plot_process,
            args=(f"Data Trend for {lg_path_str}", "Time (UTC DD/HH)", "Value",
                  raw_times_dt, raw_values.tolist(),
                  trend_times_dt, trend_values,
                  None, None, upper_warn, lower_warn)
        )
        p.start()

    @staticmethod
    def plot_data(training_set: TrainingSet, mnemonic_id: str):
        """
        Simple raw data visualization handler.

        Args:
            training_set (TrainingSet): Raw data.
            mnemonic_id (str): Mnemonic label.
        """
        if not training_set or training_set.inputs.size == 0:
            return
            
        times = training_set.inputs[:, 0]
        raw_times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times]
        raw_values = training_set.raw.tolist()
        
        p = multiprocessing.Process(
            target=_worker_plot_process,
            args=(f"Raw Data for {mnemonic_id}", "Time (UTC DD/HH)", "Value",
                  raw_times, raw_values,
                  [], [], None, None, None, None)
        )
        p.start()
