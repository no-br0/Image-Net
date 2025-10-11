import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import json
import os
import math
from Config.log_dir import TELEMETRY_LOG_FOLDER

class OptimiserTelemetryViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Stratagum Optimiser Telemetry")
        self.root.configure(bg="#1e1e1e")

        self.model_name = self._resolve_model_name()
        self.df = pd.DataFrame()
        self._last_df = pd.DataFrame()
        self.slider_start = 1
        self.slider_end = 1

        self.signal_keys = []
        self.signal_axes = {}
        self.plot_initialized = False
        self._is_moving = False

        self.lock_start = tk.BooleanVar(value=False)
        self.lock_end = tk.BooleanVar(value=True)
        self.canvas_frame = tk.Frame(self.root, bg="#1e1e1e")

        self.canvas = None
        self._init_epoch_range()
        self._init_controls()
        self._configure_layout()
        self._refresh_keys_and_plot()
        self._poll_telemetry()

        self.root.bind("<Configure>", self._on_configure)
        self.root.bind("<ButtonRelease-1>", self._on_mouse_release)

    def _resolve_model_name(self):
        try:
            with open(os.path.join("outputs", "current_model_name.json"), "r") as f:
                return json.load(f).get("model_name", "nn_model")
        except Exception as e:
            print("Could not resolve model name:", e)
            return "nn_model"

    def _poll_telemetry(self):
        if self._is_moving:
            self.root.after(1000, self._poll_telemetry)
            return
        path = os.path.join(TELEMETRY_LOG_FOLDER, f"{self.model_name}_optimiser.jsonl")
        if not os.path.exists(path):
            self.root.after(1000, self._poll_telemetry)
            return

        try:
            with open(path, "r") as f:
                rows = [json.loads(line) for line in f if line.strip()]
                new_df = pd.DataFrame(rows)
        except Exception as e:
            print("Failed to load optimiser telemetry:", e)
            self.root.after(1000, self._poll_telemetry)
            return

        if not new_df.empty and "global_epoch" in new_df.columns:
            self.df = new_df
            max_epoch = int(self.df["global_epoch"].max())
            if self.slider_end < 1:
                self.slider_end = max_epoch

        self._update_plot()
        self.root.after(1000, self._poll_telemetry)


    def _init_plot(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        num_signals = len(self.signal_keys)
        cols = math.ceil(math.sqrt(num_signals))
        rows = math.ceil(num_signals / cols)

        self.fig, axes_grid = plt.subplots(rows, cols, figsize=(14, 3.5 * rows), sharex=True)
        self.fig.patch.set_facecolor("#1e1e1e")
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.06, hspace=0.35, wspace=0.3)

        axes_grid = axes_grid.flatten()
        self.signal_axes = {}

        for i, key in enumerate(self.signal_keys):
            ax = axes_grid[i]
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.set_ylabel(key, color="white")
            if i >= len(axes_grid) - cols:
                ax.set_xlabel("Epoch", color="white")
            self.signal_axes[key] = ax

        for j in range(len(self.signal_keys), len(axes_grid)):
            axes_grid[j].axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def _refresh_keys_and_plot(self):
        path = os.path.join(TELEMETRY_LOG_FOLDER, f"{self.model_name}_optimiser.jsonl")
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                rows = [json.loads(line) for line in f if line.strip()]
                new_df = pd.DataFrame(rows)
        except Exception as e:
            print("Failed to refresh telemetry:", e)
            return

        if new_df.empty or "global_epoch" not in new_df.columns:
            return

        latest_epoch = int(new_df["global_epoch"].max())
        previous_epoch = int(self.df["global_epoch"].max()) if not self.df.empty else -1
        
        if latest_epoch < previous_epoch:
            self.slider_start = int(new_df["global_epoch"].min())
            self.slider_end = latest_epoch
            self.entry_start.delete(0, tk.END)
            self.entry_start.insert(0, str(self.slider_start))
            self.entry_end.delete(0, tk.END)
            self.entry_end.insert(0, str(self.slider_end))

        self.df = new_df

        new_keys = [
            k for k in self.df.columns
            if k != "global_epoch" and self.df[k].dtype in [float, int]
        ]

        if new_keys != self.signal_keys or not self.plot_initialized:
            self.signal_keys = new_keys
            self._init_plot()
            self.plot_initialized = True

        self._update_plot()


    def _init_controls(self):
        self.control_frame = tk.Frame(self.root, bg="#1e1e1e")

    def _init_epoch_range(self):
        self.range_frame = tk.Frame(self.root, bg="#1e1e1e")

        tk.Label(self.range_frame, text="Start Epoch:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
        self.entry_start = tk.Entry(self.range_frame, width=6)
        self.entry_start.pack(side=tk.LEFT)
        self.entry_start.bind("<Return>", lambda e: self._on_entry_commit())

        tk.Label(self.range_frame, text="End Epoch:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
        self.entry_end = tk.Entry(self.range_frame, width=6)
        self.entry_end.pack(side=tk.LEFT)
        self.entry_end.bind("<Return>", lambda e: self._on_entry_commit())

        tk.Checkbutton(self.range_frame, text="Lock to oldest", variable=self.lock_start,
                       fg="white", bg="#1e1e1e", selectcolor="#1e1e1e",
                       command=self._update_plot).pack(side=tk.LEFT)

        tk.Checkbutton(self.range_frame, text="Lock to newest", variable=self.lock_end,
                       fg="white", bg="#1e1e1e", selectcolor="#1e1e1e",
                       command=self._update_plot).pack(side=tk.LEFT)

        tk.Button(self.range_frame, text="Refresh", command=self._refresh_keys_and_plot,
                  fg="white", bg="#333333", activebackground="#555555").pack(side=tk.LEFT, padx=10)

    def _configure_layout(self):
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=1)

        self.range_frame.grid(row=0, column=0, sticky="ew")
        self.canvas_frame.grid(row=1, column=0, sticky="nsew")
        self.control_frame.grid(row=2, column=0, sticky="ew")

    def _on_entry_commit(self):
        self._update_plot()

    def _on_configure(self, event):
        if self.canvas:
            self.canvas.get_tk_widget().configure(state="disabled")
        self._is_moving = True

    def _on_mouse_release(self, event):
        if self._is_moving:
            self._is_moving = False
            if self.canvas:
                self.canvas.get_tk_widget().configure(state="normal")
            self._redraw_canvas()

    def _redraw_canvas(self):
        if self.canvas and not self._is_moving:
            self.root.after_idle(self.canvas.draw_idle)

    def _update_plot(self):
        if self.df.empty or "global_epoch" not in self.df.columns:
            for key, ax in self.signal_axes.items():
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f"{key} (Waiting for telemetry...)", color="white")
            self._redraw_canvas()
            return

        max_epoch = int(self.df["global_epoch"].max())
        min_epoch = int(self.df["global_epoch"].min())

        if self.lock_start.get():
            self.slider_start = min_epoch
            self.entry_start.delete(0, tk.END)
            self.entry_start.insert(0, str(min_epoch))
        else:
            try:
                self.slider_start = int(self.entry_start.get())
            except ValueError:
                self.slider_start = min_epoch

        if self.lock_end.get():
            self.slider_end = max_epoch
            self.entry_end.delete(0, tk.END)
            self.entry_end.insert(0, str(max_epoch))
        else:
            try:
                self.slider_end = int(self.entry_end.get())
            except ValueError:
                self.slider_end = max_epoch

        start_val = self.slider_start
        end_val = self.slider_end

        if start_val < 1 or start_val >= end_val or start_val > max_epoch:
            start_val = 1
        if end_val > max_epoch:
            end_val = max_epoch

        self.slider_start = start_val
        self.slider_end = end_val

        df = self.df[(self.df["global_epoch"] >= start_val) & (self.df["global_epoch"] <= end_val)]
        if df.equals(self._last_df):
            return
        self._last_df = df
        for key, ax in self.signal_axes.items():
            ax.clear()
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.set_ylabel(key, color="white")
            if key in df.columns and not df.empty:
                ax.plot(df["global_epoch"], df[key], color="#6fbff3", label=key, linewidth=1.5)
                ax.relim()
                ax.autoscale_view()

        self._redraw_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    viewer = OptimiserTelemetryViewer(root)
    root.mainloop()
