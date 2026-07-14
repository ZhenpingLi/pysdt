import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from algorithm.subsystem_output import SubsystemOutput
from algorithm.training_output import TrainingOutputData

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.subsystem_output import SubsystemOutput
from algorithm.training_output import TrainingOutputData
from algorithm.outlier import Outlier
from training import data_buffer
from sdtdb import sdt_db
from config.sdt_constants import DAY_IN_SECONDS

# NOTE: This implementation requires matplotlib and tkinter.
try:
    import matplotlib
    # Use TkAgg backend for embedding in Tkinter
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("ERROR: matplotlib or tkinter is not installed.")
    plt = None

CONTEXT = "HandleOutlierPlot"

@dataclass
class OutlierPlotData:
    label: str
    outlier_list: List[Outlier]

def _worker_plot_process(title: str, plot_data_list: List[OutlierPlotData]):
    """
    Worker function to run in a separate process.
    Creates a scrollable Tkinter window containing the Matplotlib figure.
    """
    if not plt or not plot_data_list:
        return

    num_plots = len(plot_data_list)
    
    # Calculate required height: e.g., 2 inches per plot
    fig_height = max(6, 2 * num_plots)
    
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Outlier Plot") 
    root.geometry("1000x800") 

    # Create a Main Frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)

    # Create a Canvas for scrolling
    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Add a Scrollbar to the Canvas
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configure the Canvas
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create another Frame INSIDE the Canvas
    # This frame will hold the Matplotlib widget
    plot_frame = tk.Frame(canvas)
    
    # Create a window on the canvas for the plot_frame
    # Anchor 'nw' ensures it starts at the top-left
    window_id = canvas.create_window((0, 0), window=plot_frame, anchor="nw")

    # --- Matplotlib Plotting ---
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, fig_height), sharex=True)
    fig.suptitle(title, fontsize=14)
    
    if num_plots == 1:
        axes = [axes]

    # Manual margin adjustment to fill the frame
    # Reduced left/right/top/bottom to make plots larger
    fig.subplots_adjust(hspace=0, top=0.96, bottom=0.08, left=0.05, right=0.98)

    for i, plot_data in enumerate(plot_data_list):
        ax = axes[i]
        times = [datetime.fromtimestamp(outlier['time'], tz=timezone.utc) for outlier in plot_data.outlier_list]
        values = [outlier['diff'] for outlier in plot_data.outlier_list]
        label = plot_data.label
        
        # Plot a horizontal line at y=0
        ax.axhline(0, color='black', linewidth=1.0, alpha=0.5)
        
        # Plot outliers
        ax.plot(times, values, 'ro', markersize=3)
        
        # Label inside the plot area to save space
        ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i < num_plots - 1:
            ax.tick_params(labelbottom=False)

    # Format the shared x-axis
    axes[-1].set_xlabel("Time (UTC HH:MM)")
    date_fmt = mdates.DateFormatter('%H:%M', tz=timezone.utc)
    axes[-1].xaxis.set_major_formatter(date_fmt)

    # Embed the figure in the Tkinter frame
    canvas_widget = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # --- Responsive Layout Logic ---
    def _on_frame_configure(event):
        """Reset the scroll region to encompass the inner frame"""
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event):
        """When the canvas is resized, resize the inner frame to match its width"""
        canvas.itemconfig(window_id, width=event.width)

    plot_frame.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    # Add Navigation Toolbar
    toolbar_frame = tk.Frame(root)
    toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar = NavigationToolbar2Tk(canvas_widget, toolbar_frame)
    toolbar.update()

    # Start the Tkinter loop
    root.mainloop()
    plt.close('all')


def _get_outlier_plot_data(system_output: SubsystemOutput) -> Optional[List[Dict[str, Any]]]:
    """
    Extracts outlier data from a SubsystemOutput and prepares it for plotting.
    """
    start_time = data_buffer.session_end - DAY_IN_SECONDS
    plot_data_list = []

    if not system_output or not system_output.mnemonic_output_list:
        return None

    for mnemonic_output in system_output.mnemonic_output_list:
        if mnemonic_output.outlier_list:
            filtered_outliers = [o for o in mnemonic_output.outlier_list if o['time'] >= start_time]
            if len(filtered_outliers) >= 2:
                times = [datetime.fromtimestamp(o['time'], timezone.utc) for o in filtered_outliers]
                values = [o['value'] for o in filtered_outliers]
                plot_data_list.append({
                    'times': times,
                    'values': values,
                    'label': mnemonic_output.mnemonic_id
                })
                
    return plot_data_list if plot_data_list else None


class HandleOutlierPlot:
    """
    A runnable task to handle plotting of outlier data.
    """

    def __init__(self, tokens: List[str]):
        if len(tokens) >= 1:
            self.id_list = [id_str for id_str in tokens]
        else:
            self.id_list = None

    def run(self):
        """
        Main execution method. Dispatches to the appropriate plotting method.
        """
        if self.id_list is None:
            group_list = data_buffer.data_output_map
            self.id_list = [name for name in group_list.keys()]
            self.plot_groups()
            return

        if len(self.id_list) == 1:
            target_id = self.id_list[0]
            if not sdt_db.exist(target_id):
                logging.warning(f"{CONTEXT}: {target_id} is not defined in AIMS Database")
                return
            
            if sdt_db.is_subsystem(target_id):
                self.plot_group()
            else:
                training_output = data_buffer.get_training_output_data(target_id)
                if training_output:
                    self.plot_mnemonic(training_output)
        else:
            self.plot_groups()

    def plot_mnemonic(self, training_output: TrainingOutputData):
        """
        Creates a plot for a single mnemonic.
        """
        if training_output.outlier_list and len(training_output.outlier_list) >= 2:
            plot_data = OutlierPlotData(training_output.mnemonic_id, training_output.outlier_list)
            title = f"Outlier Plot for {training_output.mnemonic_id}"
            p = multiprocessing.Process(target=_worker_plot_process, args=(title, [plot_data]))
            p.start()
        else:
            logging.info(f"{CONTEXT}: No outliers present in {training_output.mnemonic_id}")

    def plot_group(self):
        """
         Creates a plot for all mnemonics within a dataset group.
         """
        plot_data_list = []
        subsystem_output = data_buffer.get_subsystem_output(self.id_list[0])
        if not subsystem_output:
            return
        else:
            mnemonic_output_list : List[TrainingOutputData] = subsystem_output.mnemonic_output_list
            for mnemonic_output in mnemonic_output_list:
                if mnemonic_output.outlier_list is not None and len(mnemonic_output.outlier_list)>2:
                    outlier_list = [outlier for outlier in mnemonic_output.outlier_list if outlier['time']>data_buffer.session_time]
                    if len(outlier_list) > 1:
                        plot_data_list.append(OutlierPlotData(mnemonic_output.mnemonic_id, outlier_list))
        if plot_data_list:
            title = f"Outlier Plot for the {self.id_list[0]} subsystem"
            p = multiprocessing.Process(target=_worker_plot_process, args=(title, plot_data_list))
            p.start()
        else:
            logging.info(f"{CONTEXT}: No outliers for the Subsystem: {self.id_list[0]}")


    def plot_groups(self):
        """
        Creates a plot for multiple dataset groups.
        """
        plot_data_list = []
        for subsystem_name in self.id_list:
            if subsystem_name != "events":
                subsystem_output = data_buffer.get_subsystem_output(subsystem_name)
                if subsystem_output:
                    outlier_list : List[Outlier] = []
                    for mnemonic_output in subsystem_output.mnemonic_output_list:
                        if mnemonic_output.outlier_list is not None and len(mnemonic_output.outlier_list) >2:
                            outlier_list.extend([outlier for outlier in mnemonic_output.outlier_list if outlier['time']>data_buffer.session_time])
                    if len(outlier_list) > 5:
                        plot_data_list.append(OutlierPlotData(subsystem_name, outlier_list))

        if plot_data_list:
            title = "Outlier Plot for Multiple Groups"
            p = multiprocessing.Process(target=_worker_plot_process, args=(title, plot_data_list))
            p.start()
        else:
            logging.info(f"{CONTEXT}: No outliers in the specified groups.")

