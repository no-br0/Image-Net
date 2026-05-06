# time_view.pyw
import os
import json
import pandas as pd
import tkinter as tk
from tkinter import BooleanVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.file_utils import get_epoch_time_path


class EpochTimeViewer(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("Epoch Time Viewer")
		self.geometry("900x600")
		self.configure(bg="#1e1e1e")

		self.df = pd.DataFrame()
		self.slider_start = 0
		self.slider_end = 1

		self._file_pos = 0
		self._last_log_path = None

		self.stick_left_var = BooleanVar(value=False)
		self.stick_right_var = BooleanVar(value=True)

		self.visible_keys = {}
		self.lines = {}

		self._build_controls()
		self._build_plot()
		self._build_bottom_bar()

		self.after(500, self._tail_and_update)

	def _build_controls(self):
		bar = tk.Frame(self, bg="#1e1e1e")
		bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

		# Live mode toggle
		self.live_mode_var = BooleanVar(value=True)
		tk.Checkbutton(
			bar, text="Live Mode", variable=self.live_mode_var,
			fg="white", bg="#1e1e1e", selectcolor="#1e1e1e"
		).pack(side=tk.LEFT, padx=6)

		# Manual model entry
		tk.Label(bar, text="Model:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
		self.model_entry = tk.Entry(
			bar, width=20, bg="#1e1e1e", fg="white", insertbackground="white"
		)
		self.model_entry.bind("<Return>", lambda e: self._on_model_entry_commit())
		self.model_entry.pack(side=tk.LEFT, padx=6)

		tk.Label(bar, text="Start:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
		self.start_var = tk.StringVar(value="0")
		self.start_entry = tk.Entry(
			bar, width=6, textvariable=self.start_var,
			bg="#1e1e1e", fg="white", insertbackground="white"
		)
		self.start_entry.bind("<Return>", lambda e: self._on_range_commit())
		self.start_entry.pack(side=tk.LEFT, padx=6)

		tk.Label(bar, text="End:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
		self.end_var = tk.StringVar(value="1")
		self.end_entry = tk.Entry(
			bar, width=6, textvariable=self.end_var,
			bg="#1e1e1e", fg="white", insertbackground="white"
		)
		self.end_entry.bind("<Return>", lambda e: self._on_range_commit())
		self.end_entry.pack(side=tk.LEFT, padx=6)

		tk.Checkbutton(
			bar, text="Stick to start", variable=self.stick_left_var,
			fg="white", bg="#1e1e1e", selectcolor="#1e1e1e",
			command=self._on_range_changed
		).pack(side=tk.LEFT, padx=12)

		tk.Checkbutton(
			bar, text="Stick to end", variable=self.stick_right_var,
			fg="white", bg="#1e1e1e", selectcolor="#1e1e1e",
			command=self._on_range_changed
		).pack(side=tk.LEFT, padx=12)

		self.lbl_max_epoch = tk.Label(bar, text="Max: 0", fg="yellow", bg="#1e1e1e")
		self.lbl_max_epoch.pack(side=tk.LEFT, padx=12)

	def _build_plot(self):
		frame = tk.Frame(self, bg="#1e1e1e")
		frame.pack(fill=tk.BOTH, expand=True)

		self.figure = Figure(facecolor="#1e1e1e")
		self.ax = self.figure.add_subplot(111)
		self.ax.set_title("Epoch Time Breakdown", color="white")
		self.ax.set_ylabel("Seconds", color="white")
		self.ax.tick_params(colors="white")
		for spine in self.ax.spines.values():
			spine.set_color("white")
		self.ax.grid(True, color="#444444", linestyle="--", linewidth=0.5)
		self.ax.set_facecolor("#1e1e1e")

		self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
		self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

		container = tk.Frame(win, bg="#1e1e1e")
		container.pack(fill=tk.BOTH, expand=True)

		canvas = tk.Canvas(container, bg="#1e1e1e", highlightthickness=0)
		canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
		scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

		canvas.configure(yscrollcommand=scrollbar.set)

		panel = tk.Frame(canvas, bg="#1e1e1e")
		canvas.create_window((0, 0), window=panel, anchor="nw")

		def _update_scrollregion(event):
			canvas.configure(scrollregion=canvas.bbox("all"))

		panel.bind("<Configure>", _update_scrollregion)

		def _on_mousewheel(event):
			canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

		panel.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
		panel.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

		tk.Label(
			panel, text="Timing Toggles",
			fg="white", bg="#1e1e1e",
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

	def _resolve_log_path(self):
		if self.live_mode_var.get():
			return get_epoch_time_path()
		else:
			name = self.model_entry.get().strip()
			return get_epoch_time_path(name if name else None)

	def _on_model_entry_commit(self):
		self._last_log_path = None
		self._file_pos = 0
		self.df = pd.DataFrame()
		self._update_plot()
		self._tail_and_update()

	def _tail_and_update(self):
		try:
			log_path = self._resolve_log_path()

			# Detect model switch
			if log_path != self._last_log_path:
				self._last_log_path = log_path
				self._file_pos = 0
				self.df = pd.DataFrame()

				# Remove all dynamic lines
				for ln in list(self.lines.values()):
					try:
						ln.remove()
					except:
						pass

				self.lines.clear()

				if self.ax.legend_:
					self.ax.legend_.remove()

				self.ax.relim()
				self.ax.autoscale_view()
				self.canvas.draw_idle()

			if not os.path.exists(log_path):
				self.after(500, self._tail_and_update)
				return

			rows = []
			with open(log_path, "r", encoding="utf8") as f:
				# Check file size BEFORE seeking
				current_size = os.path.getsize(log_path)

				# Detect file truncation or replacement
				if current_size < self._file_pos:
					# File was truncated (new training run started)
					self._file_pos = 0
					self.df = pd.DataFrame()

					# Clear all dynamic lines and toggles
					for ln in list(self.lines.values()):
						try:
							ln.remove()
						except:
							pass

					self.lines.clear()

					if self.ax.legend_:
						self.ax.legend_.remove()

					self.ax.relim()
					self.ax.autoscale_view()
					self.canvas.draw_idle()

				f.seek(self._file_pos)

				new_lines = f.readlines()
				self._file_pos = f.tell()

			for line in new_lines:
				try:
					rows.append(json.loads(line))
				except:
					pass

			if not rows:
				self.after(500, self._tail_and_update)
				return

			new_df = pd.DataFrame(rows)
			if "global_epoch" not in new_df.columns:
				self.after(500, self._tail_and_update)
				return

			new_df["global_epoch"] = pd.to_numeric(new_df["global_epoch"], errors="coerce")
			new_df = new_df.dropna(subset=["global_epoch"])

			# flatten epoch_breakdown
			if "epoch_breakdown" in new_df.columns:
				breakdown_df = new_df["epoch_breakdown"].apply(pd.Series)
				new_df = pd.concat([new_df.drop(columns=["epoch_breakdown"]), breakdown_df], axis=1)

			if self.df.empty:
				self.df = new_df
			else:
				self.df = pd.concat([self.df, new_df], ignore_index=True)

			self.lbl_max_epoch.config(text=f"Max: {int(self.df['global_epoch'].max())}")

			timing_keys = [c for c in self.df.columns if c.endswith("_time")]

			# --- REMOVE toggles for keys that no longer exist ---
			for old_key in list(self.visible_keys.keys()):
				if old_key not in timing_keys:
					del self.visible_keys[old_key]

			# --- REMOVE line objects for keys that no longer exist ---
			for old_key in list(self.lines.keys()):
				if old_key not in timing_keys:
					try:
						self.lines[old_key].remove()
					except:
						pass
					del self.lines[old_key]

			# --- ADD toggles and lines for new keys ---
			for key in timing_keys:
				if key not in self.visible_keys:
					self.visible_keys[key] = BooleanVar(value=True)

				if key not in self.lines:
					ln, = self.ax.plot([], [], label=key)
					self.lines[key] = ln


			self._on_range_changed()

		except Exception as e:
			print("Time tail error:", e)

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

		for key, ln in self.lines.items():
			if key in df.columns and self.visible_keys[key].get():
				ln.set_data(df["global_epoch"], df[key])
			else:
				ln.set_data([], [])

		handles = []
		labels = []
		for key, ln in self.lines.items():
			if self.visible_keys[key].get():
				handles.append(ln)
				labels.append(key)

		if self.ax.legend_:
			self.ax.legend_.remove()
		if handles:
			self.ax.legend(handles, labels, facecolor="black", edgecolor="white", labelcolor="white")

		self.ax.relim()
		self.ax.autoscale_view()
		self.canvas.draw_idle()


if __name__ == "__main__":
	EpochTimeViewer().mainloop()
