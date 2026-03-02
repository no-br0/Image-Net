import os
import json
import pandas as pd
import tkinter as tk
from tkinter import BooleanVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Config.log_dir import GPU_LOG_PATH




class GPUViewer(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("GPU Telemetry Viewer")
		self.geometry("900x600")
		self.configure(bg="#1e1e1e")

		self.metric_colors = {
			"gpu_temp": "#ff6666",   # red
			"gpu_util": "#66ff66",   # green
			"vram_used": "#66ccff",  # blue
			"vram_total": "#3399ff", # darker blue
			"pool_used": "#cc99ff",  # purple
			"pool_total": "#9966ff", # darker purple
			"pool_free": "#cc66ff",  # magenta
		}


		self.df = pd.DataFrame()
		self.slider_start = 0
		self.slider_end = 1

		self.stick_left_var = BooleanVar(value=False)
		self.stick_right_var = BooleanVar(value=True)

		# key -> BooleanVar
		self.visible_keys = {}
		# key -> matplotlib line
		self.lines = {}

		self._build_controls()
		self._build_plot()
		self._build_bottom_bar()

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

		self.lbl_max_epoch = tk.Label(bar, text="Max: 0", fg="yellow", bg="#1e1e1e")
		self.lbl_max_epoch.pack(side=tk.LEFT, padx=12)

	def _axis_for_key(self, key):
		if key == "gpu_temp":
			return self.ax_temp
		if key == "gpu_util":
			return self.ax_util
		return self.ax_vram  # vram_used, vram_total, pool_*


	def _build_bottom_bar(self):
		bar = tk.Frame(self, bg="#1e1e1e", height=30)
		bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)
		bar.pack_propagate(False)

		self.dropdown_button = tk.Button(
			bar, text="Timing Lines ▼",
			bg="#1e1e1e", fg="white",
			activebackground="#2a2a2a", activeforeground="white",
			command=self._open_dropdown
		)
		self.dropdown_button.pack(side=tk.LEFT, padx=6)


	def _build_plot(self):
		frame = tk.Frame(self, bg="#1e1e1e")
		frame.pack(fill=tk.BOTH, expand=True)

		self.figure = Figure(facecolor="#1e1e1e")

		# MAIN AXIS (VRAM)
		self.ax_vram = self.figure.add_subplot(111)
		self.ax_vram.set_title("GPU Telemetry", color="white")
		self.ax_vram.set_ylabel("VRAM (MB)", color="#66ccff")
		self.ax_vram.tick_params(colors="#66ccff")
		for spine in self.ax_vram.spines.values():
			spine.set_color("white")
		self.ax_vram.grid(True, color="#444444", linestyle="--", linewidth=0.5)
		self.ax_vram.set_facecolor("#1e1e1e")

		# SECOND AXIS (TEMP)
		self.ax_temp = self.ax_vram.twinx()
		self.ax_temp.set_ylabel("Temp (°C)", color="#ff6666")
		self.ax_temp.tick_params(colors="#ff6666")

		# THIRD AXIS (UTIL)
		self.ax_util = self.ax_vram.twinx()
		self.ax_util.spines["right"].set_position(("axes", 1.05))
		self.ax_util.set_ylabel("Util (%)", color="#66ff66")
		self.ax_util.tick_params(colors="#66ff66")

		self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
		self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



	def _open_dropdown(self):
		if not self.visible_keys:
			return

		win = tk.Toplevel(self)
		win.configure(bg="#1e1e1e")
		win.overrideredirect(True)

		dropdown_height = 250
		width = 260
		x = self.dropdown_button.winfo_rootx()
		y = self.dropdown_button.winfo_rooty() - dropdown_height
		win.geometry(f"{width}x{dropdown_height}+{x}+{y}")

		# --- Scrollable container ---
		container = tk.Frame(win, bg="#1e1e1e")
		container.pack(fill=tk.BOTH, expand=True)

		canvas = tk.Canvas(
			container,
			bg="#1e1e1e",
			highlightthickness=0
		)
		canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		scrollbar = tk.Scrollbar(
			container,
			orient="vertical",
			command=canvas.yview
		)
		scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

		canvas.configure(yscrollcommand=scrollbar.set)

		# Frame INSIDE canvas
		panel = tk.Frame(canvas, bg="#1e1e1e")
		canvas.create_window((0, 0), window=panel, anchor="nw")

		# Update scrollregion whenever panel changes size
		def _update_scrollregion(event):
			canvas.configure(scrollregion=canvas.bbox("all"))

		panel.bind("<Configure>", _update_scrollregion)

		# Mousewheel scrolling
		def _on_mousewheel(event):
			canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

		panel.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
		panel.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

		# --- Dropdown content ---
		tk.Label(
			panel,
			text="Timing Toggles",
			fg="white",
			bg="#1e1e1e",
			font=("Arial", 10, "bold")
		).pack(anchor="w", padx=6, pady=(6, 4))

		for key in sorted(self.visible_keys.keys()):
			cb = tk.Checkbutton(
				panel,
				text=key,
				variable=self.visible_keys[key],
				command=self._update_plot,
				fg="white",
				bg="#1e1e1e",
				selectcolor="#1e1e1e",
				activebackground="#1e1e1e",
				activeforeground="white"
			)
			cb.pack(anchor="w", padx=6, pady=2)

		# Close when focus is lost
		win.bind("<FocusOut>", lambda e: win.destroy())
		win.focus_set()


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
		if not os.path.exists(GPU_LOG_PATH):
			self.after(500, self._tail_and_update)
			return

		rows = []
		with open(GPU_LOG_PATH, "r", encoding="utf8") as f:
			for line in f:
				try:
					rows.append(json.loads(line))
				except:
					pass

		df = pd.DataFrame(rows)
		if "global_epoch" not in df.columns:
			self.after(500, self._tail_and_update)
			return

		df["global_epoch"] = pd.to_numeric(df["global_epoch"], errors="coerce")
		df = df.dropna(subset=["global_epoch"])

		self.df = df
		self.lbl_max_epoch.config(text=f"Max: {int(df['global_epoch'].max())}")

		# detect all *_time keys
		gpu_keys = [
			c for c in df.columns
			if c.startswith("gpu_") or
			c.startswith("vram_") or
			c.startswith("pool_")
		]


		for key in gpu_keys:
			if key not in self.visible_keys:
				self.visible_keys[key] = BooleanVar(value=True)
			if key not in self.lines:
				axis = self._axis_for_key(key)
				ln, = axis.plot([], [], label=key)
				self.lines[key] = ln

		self._on_range_changed()
		self.after(500, self._tail_and_update)

	def _update_plot(self):
		if self.df.empty:
			for ln in self.lines.values():
				ln.set_data([], [])
			self.canvas.draw_idle()
			return

		df = self.df[
			(self.df["global_epoch"] >= self.slider_start) &
			(self.df["global_epoch"] <= self.slider_end)
		]

		# update line data on the correct axis
		for key, ln in self.lines.items():
			if key in df.columns and self.visible_keys[key].get():
				axis = self._axis_for_key(key)
				ln.set_data(df["global_epoch"], df[key])
			else:
				ln.set_data([], [])

		# rebuild legend on the MAIN axis (vram)
		handles = []
		labels = []
		for key, ln in self.lines.items():
			if self.visible_keys[key].get():
				handles.append(ln)
				labels.append(key)

		if hasattr(self.ax_vram, "legend_") and self.ax_vram.legend_:
			self.ax_vram.legend_.remove()
		if handles:
			self.ax_vram.legend(
				handles,
				labels,
				facecolor="black",
				edgecolor="white",
				labelcolor="white"
			)

		# autoscale all three axes
		self.ax_vram.relim()
		self.ax_vram.autoscale_view()

		self.ax_temp.relim()
		self.ax_temp.autoscale_view()

		self.ax_util.relim()
		self.ax_util.autoscale_view()

		self.canvas.draw_idle()



if __name__ == "__main__":
	GPUViewer().mainloop()
