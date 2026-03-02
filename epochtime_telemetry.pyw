import os
import json
import pandas as pd
import tkinter as tk
from tkinter import BooleanVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

LOG_PATH = "telemetry/epoch_times.jsonl"   # adjust to your path

class EpochTimeViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Epoch Time Viewer")
        self.geometry("900x600")
        self.configure(bg="#1e1e1e")

        self.df = pd.DataFrame()
        self.slider_start = 0
        self.slider_end = 1

        self.stick_left_var = BooleanVar(value=False)
        self.stick_right_var = BooleanVar(value=True)
        self.show_line_var = BooleanVar(value=True)

        self._build_controls()
        self._build_plot()

        self.after(500, self._tail_and_update)

    def _build_controls(self):
        bar = tk.Frame(self, bg="#1e1e1e")
        bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        tk.Label(bar, text="Start:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
        self.start_var = tk.StringVar(value="0")
        self.start_entry = tk.Entry(bar, width=6, textvariable=self.start_var,
                                    bg="#1e1e1e", fg="white", insertbackground="white")
        self.start_entry.bind("<Return>", lambda e: self._on_range_commit())
        self.start_entry.pack(side=tk.LEFT, padx=6)

        tk.Label(bar, text="End:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
        self.end_var = tk.StringVar(value="1")
        self.end_entry = tk.Entry(bar, width=6, textvariable=self.end_var,
                                  bg="#1e1e1e", fg="white", insertbackground="white")
        self.end_entry.bind("<Return>", lambda e: self._on_range_commit())
        self.end_entry.pack(side=tk.LEFT, padx=6)

        tk.Checkbutton(bar, text="Stick to start", variable=self.stick_left_var,
                       fg="white", bg="#1e1e1e", selectcolor="#1e1e1e",
                       command=self._on_range_changed).pack(side=tk.LEFT, padx=12)

        tk.Checkbutton(bar, text="Stick to end", variable=self.stick_right_var,
                       fg="white", bg="#1e1e1e", selectcolor="#1e1e1e",
                       command=self._on_range_changed).pack(side=tk.LEFT, padx=12)

        tk.Checkbutton(bar, text="Show Line", variable=self.show_line_var,
                       fg="white", bg="#1e1e1e", selectcolor="#1e1e1e",
                       command=self._update_plot).pack(side=tk.LEFT, padx=12)

        self.lbl_max_epoch = tk.Label(bar, text="Max: 0", fg="yellow", bg="#1e1e1e")
        self.lbl_max_epoch.pack(side=tk.LEFT, padx=12)

    def _build_plot(self):
        frame = tk.Frame(self, bg="#1e1e1e")
        frame.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(facecolor="#1e1e1e")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Epoch Time", color="white")
        self.ax.set_ylabel("Seconds", color="white")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")
        self.ax.grid(True, color="#444444", linestyle="--", linewidth=0.5)
        self.ax.set_facecolor("#1e1e1e")

        self.line, = self.ax.plot([], [], color="#66ccff", label="Iteration Time")
        self.ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_range_commit(self):
        try:
            self.slider_start = int(self.start_var.get())
            self.slider_end = int(self.end_var.get())
            self._on_range_changed()
        except ValueError:
            print("Invalid range")

    def _on_range_changed(self):
        if self.df.empty:
            return
        max_epoch = int(self.df["global_epoch"].max())
        min_epoch = int(self.df["global_epoch"].min())

        if self.stick_right_var.get():
            self.slider_end = max_epoch
            self.end_var.set(str(max_epoch))

        if self.stick_left_var.get():
            self.slider_start = min_epoch
            self.start_var.set(str(min_epoch))

        self._update_plot()

    def _tail_and_update(self):
        if not os.path.exists(LOG_PATH):
            self.after(500, self._tail_and_update)
            return

        rows = []
        with open(LOG_PATH, "r") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except:
                    pass

        df = pd.DataFrame(rows)
        if "global_epoch" not in df.columns or "iteration_time" not in df.columns:
            self.after(500, self._tail_and_update)
            return

        df["global_epoch"] = pd.to_numeric(df["global_epoch"], errors="coerce")
        df = df.dropna(subset=["global_epoch"])

        self.df = df
        self.lbl_max_epoch.config(text=f"Max: {int(df['global_epoch'].max())}")

        self._on_range_changed()
        self.after(500, self._tail_and_update)

    def _update_plot(self):
        if self.df.empty:
            self.line.set_data([], [])
            self.canvas.draw_idle()
            return

        df = self.df[(self.df["global_epoch"] >= self.slider_start) &
                     (self.df["global_epoch"] <= self.slider_end)]

        if self.show_line_var.get():
            self.line.set_data(df["global_epoch"], df["iteration_time"])
        else:
            self.line.set_data([], [])

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()


if __name__ == "__main__":
    EpochTimeViewer().mainloop()
