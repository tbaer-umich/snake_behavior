#!/usr/bin/env python3
"""
Snake Behavior Manual Labeling Tool - DEBUG VERSION
A GUI for manually classifying accelerometer data into behavioral categories
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime


class SnakeLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake Behavior Labeling Tool")
        self.root.geometry("1200x800")

        # Data variables
        self.data = None
        self.current_idx = 0
        self.chunk_size = 125  # samples (5 seconds at 25Hz)
        self.sampling_rate = 25  # Hz
        self.labels = []
        self.classifier_predictions = []  # Store classifier predictions
        self.classifier_loaded = False
        self.confirmed_predictions = set()
        self.data_file_path = None
        self.progress_file = None
        self.labeled_file_path = None
        self.unsaved_changes = 0
        self.plot_range = 0.25  # Default y-axis range, adjustable with zoom buttons
        self.strike_only_mode = False  # Track if we're in strike-only navigation mode
        self.strike_indices = []  # List of chunk indices that are predicted as strikes


        # Classification options
        self.behaviors = ["Still", "Locomotion", "Strike", "Uncertain"]

        self.setup_ui()
        self.setup_keybindings()

    def setup_ui(self):

        # Create main frames
        control_frame = ttk.Frame(self.root, relief='solid', borderwidth=1)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Navigation frame
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Control elements
        ttk.Button(control_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Progress", command=self.load_progress).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Now", command=self.save_labels).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="← Previous (A)", command=self.prev_chunk).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next → (D)", command=self.next_chunk).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Chunk Size (samples):").pack(side=tk.LEFT, padx=(20,5))
        self.chunk_var = tk.IntVar(value=self.chunk_size)
        chunk_spin = ttk.Spinbox(control_frame, from_=50, to=375, increment=25, 
                                width=5, textvariable=self.chunk_var, command=self.update_chunk_size)
        chunk_spin.pack(side=tk.LEFT, padx=5)

        # Load classifier button
        ttk.Button(control_frame, text="Get Classifier Predictions", 
                  command=self.load_classifier_predictions).pack(side=tk.LEFT, padx=(20,5))

        # Show strikes button (initially disabled)
        self.show_strikes_btn = ttk.Button(control_frame, text="Show Strikes Only",
                                          command=self.toggle_strike_mode, state='disabled')
        self.show_strikes_btn.pack(side=tk.LEFT, padx=5)

        # Confirm All button
        self.confirm_all_btn = ttk.Button(control_frame, text="Confirm All Predictions",
                                         command=self.confirm_all_predictions, state='disabled')
        self.confirm_all_btn.pack(side=tk.LEFT, padx=5)


        # Progress indicator
        self.progress_var = tk.StringVar(value="No data loaded")
        ttk.Label(control_frame, textvariable=self.progress_var).pack(side=tk.RIGHT, padx=5)

        # Save status indicator
        self.save_status_var = tk.StringVar(value="")
        ttk.Label(control_frame, textvariable=self.save_status_var, 
                 font=('Arial', 10), foreground='blue').pack(side=tk.RIGHT, padx=(0,20))

        # Current classification display
        self.current_class_var = tk.StringVar(value="No classification")
        class_label = ttk.Label(nav_frame, textvariable=self.current_class_var, 
                              font=('Arial', 12, 'bold'), foreground='darkgreen')
        class_label.pack(side=tk.LEFT, padx=(20, 5))

        # Zoom controls
        zoom_frame = ttk.Frame(nav_frame)
        zoom_frame.pack(side=tk.LEFT, padx=(40, 5))

        self.zoom_out_btn = ttk.Button(zoom_frame, text="Zoom Out (+)", command=self.zoom_out, width=12)
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)

        self.zoom_in_btn = ttk.Button(zoom_frame, text="Zoom In (−)", command=self.zoom_in, width=12)
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(zoom_frame, text="Show Context", command=self.show_context, width=12).pack(side=tk.LEFT, padx=2)

        # Go to chunk controls
        goto_frame = ttk.Frame(nav_frame)
        goto_frame.pack(side=tk.LEFT, padx=(20, 5))

        ttk.Label(goto_frame, text="Go to chunk:").pack(side=tk.LEFT, padx=(0, 5))
        self.goto_var = tk.StringVar()
        self.goto_entry = ttk.Entry(goto_frame, textvariable=self.goto_var, width=6)
        self.goto_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(goto_frame, text="Go", command=self.go_to_chunk, width=5).pack(side=tk.LEFT, padx=2)

        # Button section at the bottom - PACK THIS FIRST
        button_container = ttk.Frame(self.root)
        button_container.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # The actual buttons
        self.class_buttons = {}
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']

        button_row = ttk.Frame(button_container)
        button_row.pack()

        for i, (behavior, color) in enumerate(zip(self.behaviors, colors)):
           btn_frame = ttk.Frame(button_row)
           btn_frame.pack(side=tk.LEFT, padx=10)

           btn = tk.Button(btn_frame, text=behavior, bg=color,
                        width=12, height=2,
                         command=lambda b=behavior: self.classify_chunk(b))
           btn.pack(pady=2)
           self.class_buttons[behavior] = btn

           shortcut_label = tk.Label(btn_frame, text=f"Press {i+1}",
                                  font=('Arial', 8), fg='gray')
           shortcut_label.pack()

        # Plot frame - PACK THIS AFTER BUTTONS
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        self.fig.suptitle("Accelerometer Data")

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def setup_keybindings(self):
        """Setup keyboard shortcuts for fast labeling"""
        self.root.bind('<Key-1>', lambda e: self.classify_chunk("Still"))
        self.root.bind('<Key-2>', lambda e: self.classify_chunk("Locomotion"))
        self.root.bind('<Key-3>', lambda e: self.classify_chunk("Strike"))
        self.root.bind('<Key-4>', lambda e: self.classify_chunk("Uncertain"))
        self.root.bind('<Key-a>', lambda e: self.prev_chunk())
        self.root.bind('<Key-d>', lambda e: self.next_chunk())
        self.root.bind('<space>', lambda e: self.next_chunk())
        self.root.bind('<plus>', lambda e: self.zoom_out())
        self.root.bind('<minus>', lambda e: self.zoom_in())
        self.root.focus_set()

    def zoom_out(self):
        """Increase the y-axis range (zoom out)"""
        self.plot_range += 0.25
        self.update_zoom_buttons()
        self.update_display()

    def zoom_in(self):
        """Decrease the y-axis range (zoom in)"""
        if self.plot_range > 0.25:  # Don't go below minimum
            self.plot_range -= 0.25
            self.update_zoom_buttons()
            self.update_display()

    def update_zoom_buttons(self):
        """Enable/disable zoom in button based on current range"""
        if self.plot_range <= 0.25:
            self.zoom_in_btn.state(['disabled'])
        else:
            self.zoom_in_btn.state(['!disabled'])

    def show_context(self):
        """Show context window with 5 chunks before and after current chunk"""
        if self.data is None:
            return

        # Calculate chunk range (5 before, current, 5 after)
        context_before = 5
        context_after = 5
        start_chunk = max(0, self.current_idx - context_before)
        end_chunk = min(self.get_total_chunks() - 1, self.current_idx + context_after)

        # Get data indices
        start_idx = start_chunk * self.chunk_size
        end_idx = (end_chunk + 1) * self.chunk_size
        context_data = self.data.iloc[start_idx:end_idx]

        # Create popup window
        context_window = tk.Toplevel(self.root)
        context_window.title(f"Context - Chunk {self.current_idx + 1}")
        context_window.geometry("1000x600")

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create time axis centered at 0 for current chunk
        total_samples = len(context_data)
        current_chunk_start_in_context = (self.current_idx - start_chunk) * self.chunk_size
        current_chunk_center = current_chunk_start_in_context + self.chunk_size / 2

        # Time in seconds, centered at current chunk
        time = (np.arange(total_samples) - current_chunk_center) / self.sampling_rate

        # Plot all three axes with different colors
        ax.plot(time, context_data['accX'].values, label='accX', color='red', alpha=0.7, linewidth=1)
        ax.plot(time, context_data['accY'].values, label='accY', color='green', alpha=0.7, linewidth=1)
        ax.plot(time, context_data['accZ'].values, label='accZ', color='blue', alpha=0.7, linewidth=1)

        # Add vertical lines to mark chunk boundaries
        for chunk_idx in range(start_chunk, end_chunk + 1):
            chunk_start_in_context = (chunk_idx - start_chunk) * self.chunk_size
            chunk_time = (chunk_start_in_context - current_chunk_center) / self.sampling_rate

            if chunk_idx == self.current_idx:
                # Highlight current chunk
                ax.axvline(chunk_time, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Current chunk')
            else:
                ax.axvline(chunk_time, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Mark the end of the last chunk
        last_chunk_end = ((end_chunk + 1 - start_chunk) * self.chunk_size - current_chunk_center) / self.sampling_rate
        ax.axvline(last_chunk_end, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel('Time (s) - Centered on Current Chunk')
        ax.set_ylabel('Acceleration')
        ax.set_title(f'Context View - Chunks {start_chunk + 1} to {end_chunk + 1}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig, context_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def go_to_chunk(self):
        """Navigate to a specific chunk number"""
        if self.data is None:
            return

        try:
            chunk_num = int(self.goto_var.get())
            # Convert to 0-indexed
            chunk_idx = chunk_num - 1

            if 0 <= chunk_idx < self.get_total_chunks():
                self.current_idx = chunk_idx
                self.update_display()
            else:
                messagebox.showwarning("Invalid Chunk", f"Please enter a chunk number between 1 and {self.get_total_chunks()}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid chunk number")



    def load_classifier_predictions(self):
        """Load classifier and run predictions on entire dataset"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        # Select classifier JSON file
        classifier_dir = Path("./classifier")
        if not classifier_dir.exists():
            classifier_dir = Path(".")

        file_path = filedialog.askopenfilename(
            title="Select classifier JSON file",
            initialdir=classifier_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # Load the classifier stats
            with open(file_path, 'r') as f:
                stats = json.load(f)

            # Check if chunk sizes match
            if 'chunk_size' in stats:
                if stats['chunk_size'] != self.chunk_size:
                    response = messagebox.askyesno("Warning", 
                        f"Classifier expects chunk size {stats['chunk_size']} but current size is {self.chunk_size}.\n"
                        f"Continue anyway?")
                    if not response:
                        return

            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Classifying...")
            progress_window.geometry("400x120")
            progress_window.configure(bg='#e0e0e0')

            tk.Label(progress_window, text="Running classifier predictions...",
                    font=('Arial', 12)).pack(pady=(20, 10))

            progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress_bar.pack(pady=10)

            progress_text = tk.Label(progress_window, text="0 / 0 chunks",
                                     font=('Arial', 10), bg='#e0e0e0', fg='black')
            progress_text.pack(pady=5)
            progress_window.update()

            # Import and initialize classifier
            from classifier import Classifier
            classifier = Classifier(stats, self.chunk_size)

            # Calculate total chunks and batch size
            total_chunks = self.get_total_chunks()
            batch_size = 100  # Process 1000 chunks at a time

            # Initialize predictions list
            self.classifier_predictions = []
            behavior_map = {'s': 'Still', 'l': 'Locomotion', 't': 'Strike', 'u': 'Uncertain'}
            # Process in batches
            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)

                # Get data for this batch of chunks
                start_idx = batch_start * self.chunk_size
                end_idx = batch_end * self.chunk_size
                batch_data = self.data[['accX', 'accY', 'accZ']].iloc[start_idx:end_idx].copy()

                # Skip if batch is too small
                if len(batch_data) < self.chunk_size:
                    continue

                # Run classification on batch
                batch_data = batch_data.reset_index(drop=True)
                _, assigned = classifier.classify(batch_data)

                # Extract predictions from this batch
                for _, _, label, _ in assigned:
                    self.classifier_predictions.append(behavior_map.get(label, 'Uncertain'))

                # Update progress
                progress = (batch_end / total_chunks) * 100
                progress_bar['value'] = progress
                progress_text.config(text=f"{batch_end} / {total_chunks} chunks")
                progress_window.update()

            self.classifier_loaded = True

            progress_window.destroy()
            messagebox.showinfo("Success", f"Classifier predictions loaded for {len(self.classifier_predictions)} chunks")

            # Enable the show strikes button
            self.show_strikes_btn.state(['!disabled'])
            self.confirm_all_btn.state(['!disabled'])

            self.update_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load classifier predictions: {str(e)}")

    def confirm_all_predictions(self):
            """Write all unconfirmed classifier predictions to the labels list"""
            if not self.classifier_loaded:
                messagebox.showwarning("Warning", "Please load classifier predictions first.")
                return

            confirm_count = 0
            total_chunks = self.get_total_chunks()

            # Iterate through all possible chunks
            for i in range(total_chunks):
                # If the chunk is unlabeled AND we have a prediction for it
                if i < len(self.classifier_predictions) and self.labels[i] is None:
                    self.labels[i] = self.classifier_predictions[i]
                    self.confirmed_predictions.add(i)
                    self.unsaved_changes += 1
                    confirm_count += 1

            if confirm_count > 0:
                self.update_save_status()
                self.update_display()
                messagebox.showinfo("Success", f"Confirmed {confirm_count} predictions across the dataset.")

                # Auto-save after a bulk operation
                self.save_labels()
            else:
                messagebox.showinfo("Info", "No unconfirmed predictions were found.")

    def toggle_strike_mode(self):
        """Toggle between normal navigation and strike-only navigation"""
        if not self.classifier_loaded:
            return

        self.strike_only_mode = not self.strike_only_mode

        if self.strike_only_mode:
            # Build list of strike indices
            self.strike_indices = [i for i, pred in enumerate(self.classifier_predictions) 
                                  if pred == 'Strike']

            if not self.strike_indices:
                messagebox.showinfo("No Strikes", "No strikes found in classifier predictions")
                self.strike_only_mode = False
                return

            # Jump to first strike
            self.current_idx = self.strike_indices[0]
            self.show_strikes_btn.configure(text="Show All Chunks")
        else:
            # Return to normal mode
            self.show_strikes_btn.configure(text="Show Strikes Only")

        self.update_display()




    def load_progress(self):
        """Load existing labeled file and resume from first unlabeled chunk"""
        file_path = filedialog.askopenfilename(
            title="Select labeled data file to resume",
            filetypes=[("CSV files", "*_labeled.csv"), ("All CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Load the labeled data
                labeled_data = pd.read_csv(file_path)
                required_cols = ['accX', 'accY', 'accZ', 'behavior']

                if not all(col in labeled_data.columns for col in required_cols):
                    messagebox.showerror("Error", f"Data must contain columns: {required_cols}")
                    return

                # Extract the base data (without behavior column for consistency)
                self.data = labeled_data[['Date', 'Time', 'accX', 'accY', 'accZ'] if 'Date' in labeled_data.columns else ['accX', 'accY', 'accZ']].copy()

                # Set file paths
                self.labeled_file_path = Path(file_path)
                # Reconstruct original file path (remove _labeled suffix)
                file_stem = self.labeled_file_path.stem.replace('_labeled', '')
                self.data_file_path = self.labeled_file_path.parent / f"{file_stem}.csv"

                # Initialize labels array
                self.labels = [None] * self.get_total_chunks()
                self.unsaved_changes = 0

                # Map single letters back to behavior names
                behavior_map = {'s': 'Still', 'l': 'Locomotion', 't': 'Strike', 'u': 'Uncertain'}

                # Extract labels by chunk and find first unlabeled
                first_unlabeled = None
                for chunk_idx in range(self.get_total_chunks()):
                    start_idx = chunk_idx * self.chunk_size
                    end_idx = min(start_idx + self.chunk_size, len(labeled_data))
                    chunk_behaviors = labeled_data.iloc[start_idx:end_idx]['behavior'].dropna()

                    if len(chunk_behaviors) > 0 and not chunk_behaviors.isna().all():
                        # Get most common behavior in chunk and map back to full name
                        most_common = chunk_behaviors.mode().iloc[0]
                        self.labels[chunk_idx] = behavior_map.get(most_common, most_common)
                    elif first_unlabeled is None:
                        first_unlabeled = chunk_idx

                # Set current index to first unlabeled chunk, or 0 if all labeled
                self.current_idx = first_unlabeled if first_unlabeled is not None else 0

                # Count how many chunks are already labeled
                labeled_count = sum(1 for label in self.labels if label is not None)

                self.update_display()
                messagebox.showinfo("Success",
                    f"Loaded {len(self.data)} data points\n"
                    f"{labeled_count}/{self.get_total_chunks()} chunks already labeled\n"
                    f"Resuming at chunk {self.current_idx + 1}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load progress: {str(e)}")


    def prev_chunk(self):
        if self.current_idx > 0:
            if self.strike_only_mode:
                # Find previous strike
                current_pos = self.strike_indices.index(self.current_idx) if self.current_idx in self.strike_indices else -1
                if current_pos > 0:
                    self.current_idx = self.strike_indices[current_pos - 1]
            else:
                self.current_idx -= 1
            self.update_display()

    def next_chunk(self):
        """Move to next chunk, auto-confirming classifier predictions"""
        if self.strike_only_mode:
            # Find next strike
            current_pos = self.strike_indices.index(self.current_idx) if self.current_idx in self.strike_indices else -1
            if current_pos < len(self.strike_indices) - 1:
                self.current_idx = self.strike_indices[current_pos + 1]
                self.update_display()
            return

        # Normal navigation
        if self.current_idx < self.get_total_chunks() - 1:
            # Auto-confirm classifier prediction if moving forward without manual classification
            if (self.classifier_loaded and 
                self.current_idx < len(self.classifier_predictions) and
                self.labels[self.current_idx] is None):
                # Confirm the classifier prediction
                self.labels[self.current_idx] = self.classifier_predictions[self.current_idx]
                self.confirmed_predictions.add(self.current_idx)
                self.unsaved_changes += 1
                self.update_save_status()

                # Auto-save every 25 classifications
                if self.unsaved_changes >= 25:
                    self.save_labels()


            self.current_idx += 1
            self.update_display()

    def update_chunk_size(self):
        self.chunk_size = self.chunk_var.get()
        if self.data is not None:
            self.current_idx = min(self.current_idx, self.get_total_chunks() - 1)
            self.update_display()

    def load_data(self):
        """Load accelerometer data file"""
        file_path = filedialog.askopenfilename(
            title="Select accelerometer data file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                required_cols = ['accX', 'accY', 'accZ']

                if not all(col in self.data.columns for col in required_cols):
                    messagebox.showerror("Error", f"Data must contain columns: {required_cols}")
                    return

                self.data_file_path = file_path
                # Create labeled file path
                file_stem = Path(file_path).stem
                file_dir = Path(file_path).parent
                self.labeled_file_path = file_dir / f"{file_stem}_labeled.csv"

                self.current_idx = 0
                self.labels = [None] * self.get_total_chunks()
                self.unsaved_changes = 0
                self.classifier_predictions = []  # Reset predictions when loading new data
                self.confirmed_predictions = set()  # Reset confirmed predictions

                # Check if labeled file already exists and load existing labels
                if self.labeled_file_path.exists():
                    self.load_existing_labels()

                self.update_display()
                messagebox.showinfo("Success", 
                    f"Loaded {len(self.data)} data points\nWill save to: {self.labeled_file_path}")


            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def load_existing_labels(self):
        """Load existing labels from labeled file if it exists"""
        try:
            labeled_data = pd.read_csv(self.labeled_file_path)
            if 'label' in labeled_data.columns:
                # Extract labels by chunk
                for chunk_idx in range(self.get_total_chunks()):
                    start_idx = chunk_idx * self.chunk_size
                    end_idx = start_idx + self.chunk_size
                    chunk_labels = labeled_data.iloc[start_idx:end_idx]['label'].dropna()
                    if len(chunk_labels) > 0:
                        # Use most common label in chunk
                        self.labels[chunk_idx] = chunk_labels.mode().iloc[0]
                print(f"Loaded existing labels from {self.labeled_file_path}")
        except Exception as e:
            print(f"Could not load existing labels: {e}")


    def get_total_chunks(self):
        """Calculate total number of chunks in the dataset"""
        if self.data is None:
            return 0
        return len(self.data) // self.chunk_size

    def get_current_chunk(self):
        """Get data for current chunk"""
        if self.data is None:
            return None

        start_idx = self.current_idx * self.chunk_size
        end_idx = start_idx + self.chunk_size

        if start_idx >= len(self.data):
            return None

        return self.data.iloc[start_idx:end_idx]

    def update_display(self):
        """Update the plot display with current chunk"""
        if self.data is None:
            return

        chunk = self.get_current_chunk()
        if chunk is None:
            return

        # Clear previous plots
        for ax in self.axes:
            ax.clear()

        # Create time axis
        time = np.arange(len(chunk)) / self.sampling_rate

        # Plot each axis separately
        labels = ['accX', 'accY', 'accZ']
        for i, col in enumerate(labels):
            self.axes[i].plot(time, chunk[col], label=col)
            self.axes[i].legend(loc='upper right')
            self.axes[i].set_ylabel(col)
            self.axes[i].grid(True, alpha=0.3)

            # Set consistent y-axis limits: center on mean ± plot_range
            data_mean = chunk[col].mean()
            self.axes[i].set_ylim(data_mean - self.plot_range, data_mean + self.plot_range)

        # Set xlabel only on bottom plot
        self.axes[-1].set_xlabel("Time (s)")

        # Update current classification display
        current_label = self.labels[self.current_idx] if self.current_idx < len(self.labels) else None
        if current_label:
            # Check if this was a confirmed prediction or manual override
            if self.current_idx in self.confirmed_predictions:
                # This was auto-confirmed from classifier
                self.current_class_var.set(f"Current: {current_label}")
                self._set_current_class_color('darkgreen')
            else:
                # This was manually set
                if (self.classifier_loaded and 
                    self.current_idx < len(self.classifier_predictions) and
                    current_label != self.classifier_predictions[self.current_idx]):
                    # Manual override that differs from prediction
                    self.current_class_var.set(f"Current: {current_label} (revised)")
                else:
                    self.current_class_var.set(f"Current: {current_label}")
                self._set_current_class_color('darkgreen')
        elif self.classifier_loaded and self.current_idx < len(self.classifier_predictions):
            # Show classifier prediction in orange
            pred = self.classifier_predictions[self.current_idx]
            self.current_class_var.set(f"Current: {pred} (classifier prediction)")
            self._set_current_class_color('orange')
        else:
            self.current_class_var.set("Current: Unlabeled")
            self._set_current_class_color('black')

        self.fig.suptitle(f"Chunk {self.current_idx + 1}/{self.get_total_chunks()}")
        plt.tight_layout()
        self.canvas.draw()

    def _set_current_class_color(self, color):
        """Helper to set the color of the current class label"""
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label) and str(child.cget('textvariable')) == str(self.current_class_var):
                        child.configure(foreground=color)

    def classify_chunk(self, behavior):
        """Classify current chunk"""
        if self.data is None:
            return

        self.labels[self.current_idx] = behavior
        # Remove from confirmed predictions since this is manual
        self.confirmed_predictions.discard(self.current_idx)
        
        # Check if this differs from classifier prediction
        if (self.classifier_loaded and 
            self.current_idx < len(self.classifier_predictions) and
            behavior != self.classifier_predictions[self.current_idx]):
            self.current_class_var.set(f"Current: {behavior} (revised)")
        else:
            self.current_class_var.set(f"Current: {behavior}")
        self._set_current_class_color('darkgreen')

        self.unsaved_changes += 1
        self.update_save_status()

        # Auto-save every 25 classifications
        if self.unsaved_changes >= 25:
            self.save_labels()

    def save_labels(self):
        """Save current labels to labeled CSV file"""
        if self.data is None or self.labeled_file_path is None:
            messagebox.showwarning("Warning", "No data loaded to save")
            return

        try:
            # Create a copy of original data
            save_data = self.data.copy()

            # Add behavior column - expand chunk labels to individual rows
            save_data['behavior'] = None

            # Map behavior names to single letters
            behavior_map = {
                'Still': 's',
                'Locomotion': 'l',
                'Strike': 't',
                'Uncertain': 'u'  # assuming you want a letter for uncertain too
            }

            for chunk_idx, label in enumerate(self.labels):
                if label is not None:
                    start_idx = chunk_idx * self.chunk_size
                    end_idx = min(start_idx + self.chunk_size, len(save_data))
                    # Convert behavior name to single letter
                    behavior_letter = behavior_map.get(label, label)
                    save_data.loc[start_idx:end_idx-1, 'behavior'] = behavior_letter

            # Save to labeled file
            save_data.to_csv(self.labeled_file_path, index=False)

            self.unsaved_changes = 0
            self.update_save_status()
            print(f"Saved labels to {self.labeled_file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {str(e)}")

    def update_save_status(self):
        """Update the save status indicator"""
        if self.unsaved_changes == 0:
            self.save_status_var.set("All saved")
        else:
            self.save_status_var.set(f"{self.unsaved_changes} unsaved")



def main():
    root = tk.Tk()
    app = SnakeLabelingTool(root)
    # Save on exit
    def on_closing():
        if app.unsaved_changes > 0:
            if messagebox.askyesno("Unsaved Changes", 
                                 f"You have {app.unsaved_changes} unsaved changes. Save before exiting?"):
                app.save_labels()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()

if __name__ == "__main__":
    main()
