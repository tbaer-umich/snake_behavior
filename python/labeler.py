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
        self.root.focus_set()

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

            # Show loading message
            loading_window = tk.Toplevel(self.root)
            loading_window.title("Loading")
            loading_window.geometry("300x100")
            tk.Label(loading_window, text="Running classifier predictions...\nThis may take a moment.", 
                    font=('Arial', 12)).pack(pady=20)
            loading_window.update()

            # Import and initialize classifier
            from classifier import Classifier
            classifier = Classifier(stats, self.chunk_size)

            # Run classification on entire dataset
            classified_df, assigned = classifier.classify(self.data[['accX', 'accY', 'accZ']])

            # Store predictions aligned with chunks
            behavior_map = {'s': 'Still', 'l': 'Locomotion', 't': 'Strike', 'u': 'Uncertain'}
            self.classifier_predictions = [behavior_map.get(label, 'Uncertain') 
                                          for _, _, label, _ in assigned]
            self.classifier_loaded = True

            loading_window.destroy()
            messagebox.showinfo("Success", f"Classifier predictions loaded for {len(self.classifier_predictions)} chunks")
            self.update_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load classifier predictions: {str(e)}")


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
            self.current_idx -= 1
            self.update_display()

    def next_chunk(self):
        """Move to next chunk, auto-confirming classifier predictions"""
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

            # Set consistent y-axis limits: center on mean ± 0.25
            data_mean = chunk[col].mean()
            self.axes[i].set_ylim(data_mean - 0.25, data_mean + 0.25)

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
