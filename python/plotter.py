#!/usr/bin/env python3
"""
plotter.py

Plotter class that generates and saves behavior visualizations:
  - 2D cross-section plots (X-Y, Y-Z, X-Z) with chunks unconnected
  - 3D trajectory plots broken by chunk boundaries
  - Time-series line plots (for future evaluation)

Includes functionality for independent plotting of evaluated data.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utils import break_into_chunks

class Plotter:
    def __init__(self, data_file, stats, chunk_size, output_dir='./plots'):
        import pandas as pd
        self.data_file = data_file
        self.prefix = os.path.splitext(os.path.basename(data_file))[0]
        self.df = pd.read_csv(self.data_file)
        self.stats = stats
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_overall(self):
        """
        For each behavior in stats, produce and save:
          - 2D cross-section plots where chunks are not connected across boundaries
          - 3D trajectory plots split by chunk
        """
        for label, info in self.stats.items():
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
        segs_xy, times_xy = [], []
        segs_yz, times_yz = [], []
        segs_xz, times_xz = [], []
        for start, end in chunks:
            if end - start < 2:
                continue
            t = np.arange(end - start)
            xi, yi, zi = x[start:end], y[start:end], z[start:end]
            pts_xy = np.stack([xi, yi], axis=1)
            segs_xy.append(np.stack([pts_xy[:-1], pts_xy[1:]], axis=1)); times_xy.append(t[:-1])
            pts_yz = np.stack([yi, zi], axis=1)
            segs_yz.append(np.stack([pts_yz[:-1], pts_yz[1:]], axis=1)); times_yz.append(t[:-1])
            pts_xz = np.stack([xi, zi], axis=1)
            segs_xz.append(np.stack([pts_xz[:-1], pts_xz[1:]], axis=1)); times_xz.append(t[:-1])
        all_xy = np.concatenate(segs_xy, axis=0) if segs_xy else np.empty((0,2,2))
        all_tx = np.concatenate(times_xy) if times_xy else np.array([])
        all_yz = np.concatenate(segs_yz, axis=0) if segs_yz else np.empty((0,2,2))
        all_ty = np.concatenate(times_yz) if times_yz else np.array([])
        all_xz = np.concatenate(segs_xz, axis=0) if segs_xz else np.empty((0,2,2))
        all_tz = np.concatenate(times_xz) if times_xz else np.array([])
        fig, axs = plt.subplots(1,3, figsize=(18,6))
        # X-Y
        lc_xy = LineCollection(all_xy, cmap='viridis'); lc_xy.set_array(all_tx); axs[0].add_collection(lc_xy)
        axs[0].set_xlim(x.min(), x.max()); axs[0].set_ylim(y.min(), y.max())
        axs[0].set_title(f"{label.upper()} 2D X-Y")
        # Y-Z
        lc_yz = LineCollection(all_yz, cmap='viridis'); lc_yz.set_array(all_ty); axs[1].add_collection(lc_yz)
        axs[1].set_xlim(y.min(), y.max()); axs[1].set_ylim(z.min(), z.max())
        axs[1].set_title(f"{label.upper()} 2D Y-Z")
        # X-Z
        lc_xz = LineCollection(all_xz, cmap='viridis'); lc_xz.set_array(all_tz); axs[2].add_collection(lc_xz)
        axs[2].set_xlim(x.min(), x.max()); axs[2].set_ylim(z.min(), z.max())
        axs[2].set_title(f"{label.upper()} 2D X-Z")
        for ax in axs:
            ax.set_xlabel(''); ax.set_ylabel('')
        fig.colorbar(lc_xy, ax=axs[0], label='Frame')
        plt.tight_layout()
        fname = f"{self.prefix}_{label}_2d_cross.pdf"
        fig.savefig(os.path.join(self.output_dir, fname)); plt.close(fig)

    def _plot_3d(self, df_sub, label, chunks):
        x, y, z = df_sub['accX'].values, df_sub['accY'].values, df_sub['accZ'].values
        segs3d, times3d = [], []
        for start,end in chunks:
            if end - start < 2: continue
            t = np.arange(end - start)
            pts3 = np.stack([x[start:end], y[start:end], z[start:end]], axis=1)
            segs3d.append(np.stack([pts3[:-1], pts3[1:]], axis=1)); times3d.append(t[:-1])
        all_3d = np.concatenate(segs3d, axis=0) if segs3d else np.empty((0,2,3))
        all_t3 = np.concatenate(times3d) if times3d else np.array([])
        fig = plt.figure(figsize=(10,8)); ax = fig.add_subplot(111, projection='3d')
        lc3 = Line3DCollection(all_3d, cmap='viridis'); lc3.set_array(all_t3); ax.add_collection(lc3)
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max()); ax.set_zlim(z.min(), z.max())
        ax.set_title(f"{label.upper()} 3D Trajectory")
        fig.colorbar(lc3, ax=ax, label='Frame')
        fname = f"{self.prefix}_{label}_3d.pdf"
        fig.savefig(os.path.join(self.output_dir, fname)); plt.close(fig)

    def plot_time_series(self, df_sub, label):
        t = np.arange(len(df_sub))
        fig, ax = plt.subplots()
        ax.plot(t, df_sub['accX'], label='accX')
        ax.plot(t, df_sub['accY'], label='accY')
        ax.plot(t, df_sub['accZ'], label='accZ')
        ax.set_title(f"{label.upper()} Time Series")
        ax.legend()
        fname = f"{self.prefix}_{label}_timeseries.pdf"
        fig.savefig(os.path.join(self.output_dir, fname)); plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot evaluated data CSV by behavior')
    parser.add_argument('-e', '--eval-csv', required=True, help='evaluated data CSV with Behavior column')
    parser.add_argument('-c', '--chunksize', type=int, default=20, help='samples per chunk')
    parser.add_argument('-o', '--output-dir', default='plots', help='directory to save plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable debug logs')
    args = parser.parse_args()
    # Build minimal stats mapping
    import pandas as pd
    df = pd.read_csv(args.eval_csv)
    stats = {lbl: {'behavior': lbl} for lbl in df['Behavior'].unique()}
    plotter = Plotter(args.eval_csv, stats, args.chunksize, output_dir=args.output_dir)
    plotter.plot_overall()

