#!/usr/bin/env python3
"""
plotter.py

Plotter class that generates and saves behavior visualizations:
  - 2D cross-section plots (X-Y, Y-Z, X-Z) with chunks unconnected
  - 3D trajectory plots broken by chunk boundaries
  - Time-series line plots (for future evaluation)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utils import break_into_chunks

class Plotter:
    def __init__(self, training_file, stats, chunk_size, output_dir='../plots'):
        import pandas as pd
        self.df = pd.read_csv(training_file)
        self.stats = stats
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_overall(self):
        """
        For each behavior in stats, produce and save:
          - 2D cross-section plots where chunks are not connected across boundaries
          - 3D trajectory plots split by chunk
        """
        for label, info in self.stats.items():
            name = info['behavior']
            df_sub = self.df[self.df['Behavior'] == label]
            if df_sub.empty:
                continue
            chunks = break_into_chunks(df_sub, self.chunk_size)
            self._plot_2d_cross(df_sub, label, chunks)
            self._plot_3d(df_sub, label, chunks)

    def _plot_2d_cross(self, df_sub, label, chunks):
        x = df_sub['accX'].values
        y = df_sub['accY'].values
        z = df_sub['accZ'].values

        # Prepare segments and time per chunk
        segs_xy, times_xy = [], []
        segs_yz, times_yz = [], []
        segs_xz, times_xz = [], []
        for start, end in chunks:
            if end - start < 2:
                continue
            t_chunk = np.arange(end - start)
            x_chunk = x[start:end]; y_chunk = y[start:end]; z_chunk = z[start:end]
            pts_xy = np.stack([x_chunk, y_chunk], axis=1)
            segs_xy.append(np.stack([pts_xy[:-1], pts_xy[1:]], axis=1))
            times_xy.append(t_chunk)
            pts_yz = np.stack([y_chunk, z_chunk], axis=1)
            segs_yz.append(np.stack([pts_yz[:-1], pts_yz[1:]], axis=1))
            times_yz.append(t_chunk)
            pts_xz = np.stack([x_chunk, z_chunk], axis=1)
            segs_xz.append(np.stack([pts_xz[:-1], pts_xz[1:]], axis=1))
            times_xz.append(t_chunk)

        # Concatenate all chunk segments
        all_segs_xy = np.concatenate(segs_xy, axis=0) if segs_xy else np.empty((0,2,2))
        all_times_xy = np.concatenate(times_xy) if times_xy else np.array([])
        all_segs_yz = np.concatenate(segs_yz, axis=0) if segs_yz else np.empty((0,2,2))
        all_times_yz = np.concatenate(times_yz) if times_yz else np.array([])
        all_segs_xz = np.concatenate(segs_xz, axis=0) if segs_xz else np.empty((0,2,2))
        all_times_xz = np.concatenate(times_xz) if times_xz else np.array([])

        fig, axs = plt.subplots(1,3, figsize=(18,6))
        # XY plot
        lc_xy = LineCollection(all_segs_xy, cmap='viridis')
        lc_xy.set_array(all_times_xy); axs[0].add_collection(lc_xy)
        axs[0].set_xlim(x.min(), x.max()); axs[0].set_ylim(y.min(), y.max())
        axs[0].set_title(f"{label} 2D X-Y")
        # YZ plot
        lc_yz = LineCollection(all_segs_yz, cmap='viridis')
        lc_yz.set_array(all_times_yz); axs[1].add_collection(lc_yz)
        axs[1].set_xlim(y.min(), y.max()); axs[1].set_ylim(z.min(), z.max())
        axs[1].set_title(f"{label} 2D Y-Z")
        # XZ plot
        lc_xz = LineCollection(all_segs_xz, cmap='viridis')
        lc_xz.set_array(all_times_xz); axs[2].add_collection(lc_xz)
        axs[2].set_xlim(x.min(), x.max()); axs[2].set_ylim(z.min(), z.max())
        axs[2].set_title(f"{label} 2D X-Z")

        for ax in axs:
            ax.set_xlabel(''); ax.set_ylabel('')
        fig.colorbar(lc_xy, ax=axs[0], label='Frame')
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f'{label}_2d_cross.pdf'))
        plt.close(fig)

    def _plot_3d(self, df_sub, label, chunks):
        x = df_sub['accX'].values
        y = df_sub['accY'].values
        z = df_sub['accZ'].values

        segs3d_list, times3d_list = [], []
        for start, end in chunks:
            if end - start < 2:
                continue
            t_chunk = np.arange(end - start)
            pts3d = np.stack([x[start:end], y[start:end], z[start:end]], axis=1)
            segs3d_list.append(np.stack([pts3d[:-1], pts3d[1:]], axis=1))
            times3d_list.append(t_chunk)
        all_segs3d = np.concatenate(segs3d_list, axis=0) if segs3d_list else np.empty((0,2,3))
        all_times3d = np.concatenate(times3d_list) if times3d_list else np.array([])

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        lc3d = Line3DCollection(all_segs3d, cmap='viridis')
        lc3d.set_array(all_times3d); ax.add_collection(lc3d)
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max()); ax.set_zlim(z.min(), z.max())
        ax.set_title(f"{label} 3D Trajectory")
        fig.colorbar(lc3d, ax=ax, label='Frame')
        fig.savefig(os.path.join(self.output_dir, f'{label}_3d.pdf'))
        plt.close(fig)

    def plot_time_series(self, df_sub, label):
        """Saves a plain time-series plot of accX, accY, accZ vs frame."""
        t = np.arange(len(df_sub))
        fig, ax = plt.subplots()
        ax.plot(t, df_sub['accX'], label='accX')
        ax.plot(t, df_sub['accY'], label='accY')
        ax.plot(t, df_sub['accZ'], label='accZ')
        ax.set_title(f"{label} Time Series")
        ax.set_xlabel('Frame'); ax.set_ylabel('Acceleration')
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, f'{label}_timeseries.pdf'))
        plt.close(fig)

