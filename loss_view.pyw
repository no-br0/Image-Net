#loss_view.pyw
import os
import json
import pandas as pd
import tkinter as tk
from tkinter import BooleanVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.file_utils import get_loss_path

class PlotPanel(tk.Frame):
	def __init__(self, parent, title="", ylabel=""):
		super().__init__(parent, bg="#1e1e1e")
		self.figure = Figure(facecolor="#1e1e1e")#figsize=(6, 4),
		self.ax = self.figure.add_subplot(111)
		self.canvas = FigureCanvasTkAgg(self.figure, master=self)
		self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
		self.canvas.draw()

		self.bind("<Configure>", self._on_configure)
		self._resize_pending = False
		self.bind("<ButtonRelease-1>", self._on_mouse_release)


		self.ax.set_title(title, color="white", fontsize=10)
		self.ax.set_ylabel(ylabel, color="white")
		self.ax.tick_params(colors="white")
		for spine in self.ax.spines.values():
			spine.set_color("white")
		self.ax.xaxis.label.set_color("white")
		self.ax.yaxis.label.set_color("white")
		self.ax.grid(True, color="#444444", linestyle="--", linewidth=0.5)
		self.ax.set_facecolor("#1e1e1e")

	def draw_idle(self):
		self.canvas.draw_idle()
		
	def _on_configure(self, event):
		self._resize_pending = True
		
	def _on_mouse_release(self, event):
		for panel in [self.panel_loss, self.panel_deriv, self.panel_break, self.panel_acc]:
			if getattr(panel, "_resize_pending", False):
				panel._resize_pending = False
				panel.canvas.draw_idle()





class TelemetryViewer(tk.Tk):
	def _on_range_changed(self):
		self._sync_epoch_range()
		self._update_master()
		self._update_derivatives()
		self._update_breakdown()
		self._update_accuracy()

	def __init__(self):
		super().__init__()
		self.title("Telemetry Viewer")
		self.geometry("1400x900")
		self.configure(bg="#1e1e1e")

		self.minsize(900,600)
		
		self.grid_columnconfigure(0, weight=1)
		self.grid_rowconfigure(0, weight=0)
		self.grid_rowconfigure(1, weight=1)
		self.grid_rowconfigure(2, weight=0)

		self.df = pd.DataFrame()
		self.slider_start = 0
		self.slider_end = 1
		self._last_epoch = -1

		# NEW: tailing + legend state
		self._file_pos = 0
		self._last_show_legends = True
		self._last_deriv_show_legends = True
		self._last_break_show_legends = True
		self._last_acc_show_legends = True
		self._last_deriv_visible = set()
		self._last_break_visible = set()
		self._last_acc_visible = set()

		self.visible_break_keys = {}
		self.visible_deriv_keys = {}
		self.visible_acc_keys = {}

		self._last_break_keys = set()
		self._last_deriv_keys = set()
		self._last_acc_keys = set()

		self._break_checkbuttons = {}
		self._deriv_checkbuttons = {}
		self._acc_checkbuttons = {}

		self.show_legends_var = BooleanVar(value=True)

		# Top control bar
		top_controls = tk.Frame(self, bg="#1e1e1e")
		#top_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
		top_controls.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
		
		self.live_mode_var = BooleanVar(value=True)
		tk.Checkbutton(top_controls, text="Live Mode", variable=self.live_mode_var,
					   fg="white", bg="#1e1e1e", selectcolor="#1e1e1e").pack(side=tk.LEFT, padx=6)

		tk.Label(top_controls, text="Model:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
		self.model_entry = tk.Entry(top_controls, width=20, bg="#1e1e1e", fg="white", insertbackground="white")
		self.model_entry.bind("<Return>", lambda e: self._on_model_entry_commit())
		self.model_entry.pack(side=tk.LEFT, padx=6)

		tk.Label(top_controls, text="Start:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
		self.start_var = tk.StringVar(value="0")
		self.start_entry = tk.Entry(top_controls, width=6, textvariable=self.start_var, bg="#1e1e1e", fg="white", insertbackground="white")
		self.start_entry.bind("<Return>", lambda e: self._on_epoch_entry_commit())
		self.start_entry.pack(side=tk.LEFT, padx=6)

		tk.Label(top_controls, text="End:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
		self.end_var = tk.StringVar()
		self.end_entry = tk.Entry(top_controls, width=6, textvariable=self.end_var, bg="#1e1e1e", fg="white", insertbackground="white")
		self.end_entry.bind("<Return>", lambda e: self._on_epoch_entry_commit())
		self.end_entry.pack(side=tk.LEFT, padx=6)

		self.lbl_max_epoch = tk.Label(top_controls, text="Max: 0", fg="yellow", bg="#1e1e1e", font=("Arial", 10, "bold"))
		self.lbl_max_epoch.pack(side=tk.LEFT, padx=12)

		self.stick_right_var = BooleanVar(value=True)
		tk.Checkbutton(top_controls, text="Stick to end", variable=self.stick_right_var,
					   fg="white", bg="#1e1e1e", selectcolor="#1e1e1e", command=self._on_range_changed).pack(side=tk.LEFT, padx=12)

		self.stick_left_var = BooleanVar(value=False)
		tk.Checkbutton(top_controls, text="Stick to start", variable=self.stick_left_var,
					   fg="white", bg="#1e1e1e", selectcolor="#1e1e1e", command=self._on_range_changed).pack(side=tk.LEFT, padx=12)

		tk.Checkbutton(top_controls, text="Show Legends", variable=self.show_legends_var,
					   fg="white", bg="#1e1e1e", selectcolor="#1e1e1e", command=self._toggle_legends).pack(side=tk.LEFT, padx=12)

		# Plot panels
		grid_frame = tk.Frame(self, bg="#1e1e1e")
		#grid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
		grid_frame.grid(row=1, column=0, sticky="nsew")

		for r in range(4):
			grid_frame.grid_rowconfigure(r, weight=1)
		for c in range(2):
			grid_frame.grid_columnconfigure(c, weight=1)

		self.panel_loss = PlotPanel(grid_frame, title="Total Raw Loss", ylabel="Loss")
		self.panel_deriv = PlotPanel(grid_frame, title="Raw Loss Derivatives", ylabel="Value")
		self.panel_break = PlotPanel(grid_frame, title="Raw Loss Breakdown", ylabel="Component value")
		self.panel_acc = PlotPanel(grid_frame, title="Accuracy", ylabel="Accuracy")

		self.panel_loss.grid(row=0, column=0, sticky="nsew")
		self.panel_deriv.grid(row=0, column=1, sticky="nsew")
		self.panel_break.grid(row=1, column=0, sticky="nsew")
		self.panel_acc.grid(row=1, column=1, sticky="nsew")

		# Bottom taskbar
		self.dropdown_bar = tk.Frame(self, bg="#1e1e1e", height=30)
		self.dropdown_bar.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
		#self.dropdown_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)
		#self.dropdown_bar.pack_propagate(False)

		self.break_button = tk.Button(self.dropdown_bar, text="Breakdown ▼", bg="#1e1e1e", fg="white", activebackground="#2a2a2a", activeforeground="white", command=self._open_break_dropdown)
		self.deriv_button = tk.Button(self.dropdown_bar, text="Derivatives ▼", bg="#1e1e1e", fg="white", activebackground="#2a2a2a", activeforeground="white", command=self._open_deriv_dropdown)
		self.acc_button = tk.Button(self.dropdown_bar, text="Accuracy ▼", bg="#1e1e1e", fg="white", activebackground="#2a2a2a", activeforeground="white", command=self._open_acc_dropdown)

		self.break_button.pack(side=tk.LEFT, padx=6)
		self.deriv_button.pack(side=tk.LEFT, padx=6)
		self.acc_button.pack(side=tk.LEFT, padx=6)
		
		
	def _redraw_all(self):
		self.panel_loss.draw_idle()
		self.panel_deriv.draw_idle()
		self.panel_break.draw_idle()
		self.panel_acc.draw_idle()


	def _open_dropdown(self, button, keys_dict, update_callback, title):
		win = tk.Toplevel(self)
		win.configure(bg="#1e1e1e")
		win.overrideredirect(True)

		
		# Estimate dropdown height
		dropdown_height = 200
		width = 220
		x = button.winfo_rootx()
		y = button.winfo_rooty() - dropdown_height
		win.geometry(f"{width}x{dropdown_height}+{x}+{y}")
		win.minsize(width, dropdown_height)
		win.maxsize(width, dropdown_height)

		# Scrollable canvas
		canvas = tk.Canvas(win, bg="#1e1e1e", highlightthickness=0, height=dropdown_height)
		canvas.grid(row=0, column=0, sticky="ns")
		#canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
		#scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		
		def _on_mousewheel(event):
			canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

		canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
		canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))


		panel = tk.Frame(canvas, bg="#1e1e1e")
		canvas.create_window((0, 0), window=panel, anchor="nw")
		panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

		# Label at top
		tk.Label(panel, text=title, fg="white", bg="#1e1e1e", font=("Arial", 10, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
		#tk.Label(win, text=title, fg="white", bg="#1e1e1e", font=("Arial", 10, "bold")).grid(row=0, sticky="nw")

		# Checkbuttons
		for key in sorted(keys_dict):
			cb = tk.Checkbutton(panel, text=key, variable=keys_dict[key],
								command=update_callback, fg="white", bg="#1e1e1e", selectcolor="#1e1e1e")
			cb.pack(anchor="w", padx=6, pady=2)
		
		# Close on focus out
		win.bind("<FocusOut>", lambda e: win.destroy())
		win.focus_set()

	def _open_break_dropdown(self):
		self._open_dropdown(self.break_button, self.visible_break_keys, lambda:(self._update_breakdown(), 
																				self.panel_break.ax.relim(),
																				self.panel_break.ax.autoscale_view(),
																				self.panel_break.draw_idle(),
																				), "Breakdown Toggles")

	def _open_deriv_dropdown(self):
		self._open_dropdown(self.deriv_button, self.visible_deriv_keys, lambda:(self._update_derivatives(),
																				self.panel_deriv.ax.relim(),	
																				self.panel_deriv.ax.autoscale_view(), 
																				self.panel_deriv.draw_idle(),
																				), "Derivative Toggles")

	def _open_acc_dropdown(self):
		self._open_dropdown(self.acc_button, self.visible_acc_keys, lambda:(self._update_accuracy(), 
																			self.panel_acc.ax.relim(),
																			self.panel_acc.ax.autoscale_view(),
																			self.panel_acc.draw_idle(),
																			), "Accuracy Toggles")

	def _sync_epoch_range(self):
		if self.df.empty:
			return
		max_epoch = int(self.df["global_epoch"].max())
		min_epoch = int(self.df["global_epoch"].min())
		self.lbl_max_epoch.config(text=f"Max: {max_epoch}")
		if self.stick_right_var.get():
			self.slider_end = max_epoch
			self.end_var.set(str(self.slider_end))
		if self.stick_left_var.get():
			self.slider_start = min_epoch
			self.start_var.set(str(self.slider_start))

	def _update_master(self):
		if self.df.empty or "global_epoch" not in self.df.columns or "total_raw_loss" not in self.df.columns:
			self.line_raw_loss.set_xdata([])
			self.line_raw_loss.set_ydata([])
			self.min_loss_dot.set_xdata([])
			self.min_loss_dot.set_ydata([])
			return

		df = self.df[(self.df["global_epoch"] >= self.slider_start) & (self.df["global_epoch"] <= self.slider_end)]
		self.line_raw_loss.set_xdata(df["global_epoch"])
		self.line_raw_loss.set_ydata(df["total_raw_loss"])

		if not df.empty:
			min_idx = df["total_raw_loss"].idxmin()
			min_epoch = df.loc[min_idx, "global_epoch"]
			min_value = df.loc[min_idx, "total_raw_loss"]
			self.min_loss_dot.set_xdata([min_epoch])
			self.min_loss_dot.set_ydata([min_value])
		else:
			self.min_loss_dot.set_xdata([])
			self.min_loss_dot.set_ydata([])

		self.panel_loss.ax.relim()
		self.panel_loss.ax.autoscale_view()

		show = self.show_legends_var.get()

		if show and (show != self._last_show_legends):
			handles = [self.line_raw_loss, self.min_loss_dot]
			labels = [ln.get_label() for ln in handles]
			if self.panel_loss.ax.legend_:
				self.panel_loss.ax.legend_.remove()
			self.panel_loss.ax.legend(handles, labels, facecolor="black", edgecolor="white", labelcolor="white")
		elif not show and self.panel_loss.ax.legend_ and show != self._last_show_legends:
			self.panel_loss.ax.legend_.remove()

		self._last_show_legends = show



	def _init_plot_lines(self):
		self.line_raw_loss, = self.panel_loss.ax.plot([], [], label="Total Raw Loss", color="#66ccff")
		self.min_loss_dot, = self.panel_loss.ax.plot([], [], 'o', color='red', markersize=6, label="Min Loss")
		self.panel_loss.ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

		self.deriv_lines = {}
		self.deriv_map = {
			"Δ raw Loss": "raw_loss_delta",
			"Δ² raw Loss": "raw_loss_curvature",
			"|Δ raw Loss|": "abs_raw_loss_delta",
			"|Δ² raw Loss|": "abs_raw_loss_curvature",
			"|Δ|Δ raw loss||": "abs_delta_abs_delta_raw"
		}
		for label in self.deriv_map:
			ln, = self.panel_deriv.ax.plot([], [], label=label)
			self.deriv_lines[label] = ln
		self.panel_deriv.ax.axhline(0, color="gray", linestyle="--", linewidth=1)
		self.panel_deriv.ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

		self.break_lines = {}
		self.acc_lines = {}
		self.acc_labels = [
			("Binary Overall", ("binary_overall", None)),
			("Continuous Overall", ("continuous_overall", None)),
			("Binary R", ("binary_per_channel", 0)),
			("Binary G", ("binary_per_channel", 1)),
			("Binary B", ("binary_per_channel", 2)),
			("Continuous R", ("continuous_per_channel", 0)),
			("Continuous G", ("continuous_per_channel", 1)),
			("Continuous B", ("continuous_per_channel", 2)),
		]
		for label, _ in self.acc_labels:
			ln, = self.panel_acc.ax.plot([], [], label=label)
			self.acc_lines[label] = ln
		self.panel_acc.ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

		self.after(1000, self._tail_and_update)


	def _resolve_log_path(self):
		if self.live_mode_var.get():
			return get_loss_path()
		else:
			name = self.model_entry.get().strip()
			return get_loss_path(name if name else None)


	def _tail_and_update(self):
		try:
			# Resolve the correct telemetry file path
			log_path = self._resolve_log_path()

			# Detect model/folder change → reset everything
			if log_path != getattr(self, "_last_log_path", None):
				self._last_log_path = log_path
				self._file_pos = 0
				self.df = pd.DataFrame()
				self._last_epoch = -1

				# RAW LOSS PANEL — DO NOT REMOVE LINES
				# Only clear their data
				self.line_raw_loss.set_xdata([])
				self.line_raw_loss.set_ydata([])
				self.min_loss_dot.set_xdata([])
				self.min_loss_dot.set_ydata([])

				if self.panel_loss.ax.legend_:
					self.panel_loss.ax.legend_.remove()

				self.panel_loss.ax.relim()
				self.panel_loss.ax.autoscale_view()
				self.panel_loss.draw_idle()

				# OTHER PANELS — REMOVE LINES COMPLETELY
				for panel, lines, keys, checkbuttons in [
					(self.panel_break, self.break_lines, self.visible_break_keys, self._break_checkbuttons),
					(self.panel_deriv, self.deriv_lines, self.visible_deriv_keys, self._deriv_checkbuttons),
					(self.panel_acc, self.acc_lines, self.visible_acc_keys, self._acc_checkbuttons),
				]:
					# Remove line objects from axes
					for ln in list(lines.values()):
						try:
							ln.remove()
						except:
							pass

					# Clear dictionaries
					lines.clear()
					checkbuttons.clear()

					# Remove legend if present
					if panel.ax.legend_:
						panel.ax.legend_.remove()

					# Reset axes
					panel.ax.relim()
					panel.ax.autoscale_view()
					panel.draw_idle()

				# Reset legend state flags
				self._last_show_legends = True
				self._last_deriv_show_legends = True
				self._last_break_show_legends = True
				self._last_acc_show_legends = True
				self._last_deriv_visible = set()
				self._last_break_visible = set()
				self._last_acc_visible = set()
				self._last_break_keys.clear()
				self._last_deriv_keys.clear()
				self._last_acc_keys.clear()


			# If the telemetry file doesn't exist yet → wait
			if not os.path.exists(log_path):
				self.after(1000, self._tail_and_update)
				return

			# Detect file truncation or reset
			file_size = os.path.getsize(log_path)
			if not hasattr(self, "_file_pos") or file_size < self._file_pos:
				self._file_pos = 0
				self.df = pd.DataFrame()

			# Read new lines
			new_rows = []
			with open(log_path, "r") as f:
				f.seek(self._file_pos)
				new_lines = f.readlines()
				self._file_pos = f.tell()

			for line in new_lines:
				try:
					new_rows.append(json.loads(line))
				except:
					pass

			# Append new data
			new_df = None
			if new_rows:
				new_df = pd.DataFrame(new_rows)
				if "global_epoch" in new_df.columns:
					new_df["global_epoch"] = pd.to_numeric(new_df["global_epoch"], errors="coerce")
					new_df = new_df.dropna(subset=["global_epoch"])

				if self.df.empty:
					self.df = new_df
				else:
					self.df = pd.concat([self.df, new_df], ignore_index=True)

			# If still empty → nothing to plot
			if self.df.empty:
				self.after(1000, self._tail_and_update)
				return

			# Update all plots
			self._sync_epoch_range()
			self._update_master()
			self._update_derivatives()
			self._update_breakdown(new_df["raw_breakdown"] if (new_df is not None and "raw_breakdown" in new_df.columns) else None)
			self._update_accuracy()

			self._redraw_all()

		except Exception as e:
			print("Tail error:", e)

		self.after(1000, self._tail_and_update)


	def _on_model_entry_commit(self):
		self._last_lod_path = None
		self._file_pos = 0
		self.df = pd.DataFrame()
		self._last_epoch = -1
		self._tail_and_update()


	def _on_epoch_entry_commit(self):
		try:
			self.slider_start = int(self.start_var.get())
			self.slider_end = int(self.end_var.get())
			self._on_range_changed()
		except ValueError:
			print("Invalid epoch range")


	def _update_derivatives(self):
		if self.df.empty or "global_epoch" not in self.df.columns:
			for ln in self.deriv_lines.values():
				ln.set_xdata([])
				ln.set_ydata([])
			return

		df = self.df[(self.df["global_epoch"] >= self.slider_start) & (self.df["global_epoch"] <= self.slider_end)]
		current_keys = set(self.deriv_map.keys())
		# REMOVE toggles for keys that disappeared
		for old_key in list(self.visible_deriv_keys.keys()):
			if old_key not in current_keys:
				del self.visible_deriv_keys[old_key]

		# ADD toggles for new keys
		for key in current_keys:
			if key not in self.visible_deriv_keys:
				self.visible_deriv_keys[key] = BooleanVar(value=True)

		self._last_deriv_keys = current_keys


		visible_now = set()

		for label, col in self.deriv_map.items():
			if label not in self.deriv_lines:
				ln, = self.panel_deriv.ax.plot([], [], label=label)
				self.deriv_lines[label] = ln
			if label not in self.visible_deriv_keys:
				self.visible_deriv_keys[label] = BooleanVar(value=True)

			if col in df.columns and self.visible_deriv_keys[label].get():
				self.deriv_lines[label].set_xdata(df["global_epoch"])
				self.deriv_lines[label].set_ydata(df[col])
				visible_now.add(label)
			else:
				self.deriv_lines[label].set_xdata([])
				self.deriv_lines[label].set_ydata([])

		# Legend: only rebuild if visibility or show_legends changed
		show = self.show_legends_var.get()
		if show and (visible_now != self._last_deriv_visible or show != self._last_deriv_show_legends):
			handles = [ln for lbl, ln in self.deriv_lines.items() if self.visible_deriv_keys.get(lbl, BooleanVar()).get()]
			labels = [ln.get_label() for ln in handles]
			if self.panel_deriv.ax.legend_:
				self.panel_deriv.ax.legend_.remove()
			self.panel_deriv.ax.legend(handles, labels, facecolor="black", edgecolor="white", labelcolor="white")
		elif not show and self.panel_deriv.ax.legend_ and show != self._last_deriv_show_legends:
			self.panel_deriv.ax.legend_.remove()

		self._last_deriv_visible = visible_now
		self._last_deriv_show_legends = show

		self.panel_deriv.ax.relim()
		self.panel_deriv.ax.autoscale_view()


	def _update_breakdown(self, new_breakdown_rows=None):
		if self.df.empty or "global_epoch" not in self.df.columns or "raw_breakdown" not in self.df.columns:
			for ln in self.break_lines.values():
				ln.set_xdata([])
				ln.set_ydata([])
			return

		df = self.df[(self.df["global_epoch"] >= self.slider_start) & (self.df["global_epoch"] <= self.slider_end)]

		breakdown_keys = set(self._last_break_keys)
		if new_breakdown_rows is not None:
			for rb in new_breakdown_rows:
				if isinstance(rb, dict):
					breakdown_keys.update(rb.keys())
		else:
			# Fallback full scan if needed
			for rb in df["raw_breakdown"]:
				if isinstance(rb, dict):
					breakdown_keys.update(rb.keys())

		# REMOVE toggles for keys that disappeared
		for old_key in list(self.visible_break_keys.keys()):
			if old_key not in breakdown_keys:
				del self.visible_break_keys[old_key]

		# ADD toggles for new keys
		for key in breakdown_keys:
			if key not in self.visible_break_keys:
				self.visible_break_keys[key] = BooleanVar(value=True)

		self._last_break_keys = breakdown_keys

		visible_now = set()

		for key in sorted(breakdown_keys):
			values = [rb.get(key, None) if isinstance(rb, dict) else None for rb in df["raw_breakdown"]]
			if key not in self.break_lines:
				ln, = self.panel_break.ax.plot([], [], label=key)
				self.break_lines[key] = ln
			if key not in self.visible_break_keys:
				self.visible_break_keys[key] = BooleanVar(value=True)

			if self.visible_break_keys[key].get():
				self.break_lines[key].set_xdata(df["global_epoch"])
				self.break_lines[key].set_ydata(values)
				visible_now.add(key)
			else:
				self.break_lines[key].set_xdata([])
				self.break_lines[key].set_ydata([])

		show = self.show_legends_var.get()
		if show and (visible_now != self._last_break_visible or show != self._last_break_show_legends):
			handles = [ln for key, ln in self.break_lines.items() if self.visible_break_keys.get(key, BooleanVar()).get()]
			labels = [ln.get_label() for ln in handles]
			if self.panel_break.ax.legend_:
				self.panel_break.ax.legend_.remove()
			self.panel_break.ax.legend(handles, labels, facecolor="black", edgecolor="white", labelcolor="white")
		elif not show and self.panel_break.ax.legend_ and show != self._last_break_show_legends:
			self.panel_break.ax.legend_.remove()

		self._last_break_visible = visible_now
		self._last_break_show_legends = show

		self.panel_break.ax.relim()
		self.panel_break.ax.autoscale_view()


	def _update_accuracy(self):
		if self.df.empty or "global_epoch" not in self.df.columns or "accuracy" not in self.df.columns:
			for ln in self.acc_lines.values():
				ln.set_xdata([])
				ln.set_ydata([])
			return

		df = self.df[(self.df["global_epoch"] >= self.slider_start) & (self.df["global_epoch"] <= self.slider_end)]
		current_keys = set(lbl for lbl, _ in self.acc_labels)
		# REMOVE toggles for keys that disappeared
		for old_key in list(self.visible_acc_keys.keys()):
			if old_key not in current_keys:
				del self.visible_acc_keys[old_key]

		# ADD toggles for new keys
		for key in current_keys:
			if key not in self.visible_acc_keys:
				self.visible_acc_keys[key] = BooleanVar(value=True)

		self._last_acc_keys = current_keys

		visible_now = set()

		for label, (key, idx) in self.acc_labels:
			if label not in self.acc_lines:
				ln, = self.panel_acc.ax.plot([], [], label=label)
				self.acc_lines[label] = ln
			if label not in self.visible_acc_keys:
				self.visible_acc_keys[label] = BooleanVar(value=True)

			values = []
			for acc in df["accuracy"]:
				if not isinstance(acc, dict):
					values.append(None)
				elif key not in acc:
					values.append(None)
				elif idx is None:
					values.append(acc[key])
				else:
					val = acc[key]
					values.append(val[idx] if isinstance(val, (list, tuple)) and len(val) > idx else None)

			if self.visible_acc_keys[label].get():
				self.acc_lines[label].set_xdata(df["global_epoch"])
				self.acc_lines[label].set_ydata(values)
				visible_now.add(label)
			else:
				self.acc_lines[label].set_xdata([])
				self.acc_lines[label].set_ydata([])

		show = self.show_legends_var.get()
		if show and (visible_now != self._last_acc_visible or show != self._last_acc_show_legends):
			handles = [ln for lbl, ln in self.acc_lines.items() if self.visible_acc_keys.get(lbl, BooleanVar()).get()]
			labels = [ln.get_label() for ln in handles]
			if self.panel_acc.ax.legend_:
				self.panel_acc.ax.legend_.remove()
			self.panel_acc.ax.legend(handles, labels, facecolor="black", edgecolor="white", labelcolor="white")
		elif not show and self.panel_acc.ax.legend_ and show != self._last_acc_show_legends:
			self.panel_acc.ax.legend_.remove()

		self._last_acc_visible = visible_now
		self._last_acc_show_legends = show

		self.panel_acc.ax.relim()
		self.panel_acc.ax.autoscale_view()


	def _toggle_legends(self):
		# Just trigger a refresh; legend logic is in update functions
		self._last_show_legends = not self.show_legends_var.get()  # force mismatch
		self._update_master()
		self._update_derivatives()
		self._update_breakdown()
		self._update_accuracy()
		self._redraw_all()


if __name__ == "__main__":
	viewer = TelemetryViewer()
	viewer._init_plot_lines()
	viewer.mainloop()
