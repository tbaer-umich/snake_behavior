#!/usr/bin/env python3
"""
evaluate.py

Evaluator class to:
  1. Load a trained classifier (JSON of behavior stats) or train via Trainer if missing
  2. Load new accelerometer CSV (Date,Time,accX,accY,accZ)
  3. Split into fixed-size chunks, compute each chunk's covariance
  4. Compute AIRM distance to each behavior mean covariance and assign labels
  5. Save classified data to CSV with new behavior column
  6. Use Plotter to generate per-behavior 2D/3D plots of classified data
  7. Optionally generate debug plots for top/bottom X% confident chunks
"""
import warnings
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

import argparse
import logging
import os
import sys
import json

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm, logm, norm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from classifier import Classifier

from utils import break_into_chunks
from train import Trainer
from plotter import Plotter

class Evaluator:
    def __init__(self,
        model_file: str,
        new_data_file: str,
        training_input: str,
        chunksize: int,
        output_csv: str,
        eval_plots_dir: str,
        verbose: bool = False,
        skip_normal_plots: bool = False,
        debug_behaviors: list[str] = None,
        debug_top_percent: float = 0.1,
        debug_bottom_percent: float = 0.1,
    ):

        self.model_file = model_file
        self.new_data_file = new_data_file
        self.training_input = training_input
        self.chunksize = chunksize
        self.output_csv = output_csv
        self.eval_plots_dir = eval_plots_dir
        self.verbose = verbose
        self.skip_normal_plots = skip_normal_plots
        self.debug_behaviors = debug_behaviors
        self.debug_top_percent = debug_top_percent
        self.debug_bottom_percent = debug_bottom_percent

        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose:
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        else:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        # Suppress verbose matlplotlib debug logs
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensure output directories
        os.makedirs(os.path.dirname(self.model_file) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv) or '.', exist_ok=True)
        os.makedirs(self.eval_plots_dir, exist_ok=True)

    def load_or_train_model(self):
        if not os.path.isfile(self.model_file):
            self.logger.info(f"No model found at {self.model_file}, training a new one.")
            trainer = Trainer(
                training_file=self.training_input,
                chunksize=self.chunksize,
                verbose=self.args.verbose
            )
            # Trainer.save_statistics writes to ./classifier/training_stats.json by default
            # TODO: as more models are trained these should have unique names
            model_path = trainer.run()
            self.model_file = model_path
        with open(self.model_file, 'r') as f:
            self.stats = json.load(f)
        # Map label -> behavior name
        self.behavior_map = {lbl: info['behavior'] for lbl,info in self.stats.items()}
        self.logger.info(f"Loaded model with behaviors: {list(self.stats.keys())}")
        self.classifier = Classifier(self.stats, self.chunksize)

    def evaluate(self):
        # Load raw data and run the shared classifier
        df = pd.read_csv(self.new_data_file)
        df, assigned = self.classifier.classify(df)

        # mirror old behavior column
        df['behavior'] = df['predictedBehavior']
        self.assigned = assigned
        self.df = df

        # persist results
        df.to_csv(self.output_csv, index=False)
        self.logger.info(f"Saved classified data to {self.output_csv}")

    def _plot_debug_chunks(self, chunks, label, which, percent):
        """
        chunks: list of (start,end,dist)
        which: 'top' or 'bottom'
        percent: fraction
        """
        df = self.df
        # Prepare segments per chunk
        segs_xy, times_xy = [], []
        segs_yz, times_yz = [], []
        segs_xz, times_xz = [], []
        segs3d, times3d = [], []
        for start,end,dist in chunks:
            block = df.iloc[start:end]
            t = np.arange(end-start)
            x = block['accX'].values; y = block['accY'].values; z = block['accZ'].values
            # 2D segments
            pts_xy = np.stack([x,y], axis=1)
            segs_xy.append(np.stack([pts_xy[:-1], pts_xy[1:]], axis=1)); times_xy.append(t[:-1])
            pts_yz = np.stack([y,z], axis=1)
            segs_yz.append(np.stack([pts_yz[:-1], pts_yz[1:]], axis=1)); times_yz.append(t[:-1])
            pts_xz = np.stack([x,z], axis=1)
            segs_xz.append(np.stack([pts_xz[:-1], pts_xz[1:]], axis=1)); times_xz.append(t[:-1])
            # 3D segments
            pts3 = np.stack([x,y,z], axis=1)
            segs3d.append(np.stack([pts3[:-1], pts3[1:]], axis=1)); times3d.append(t[:-1])

        # Concatenate
        all_xy = np.concatenate(segs_xy, axis=0) if segs_xy else np.empty((0,2,2))
        all_tx = np.concatenate(times_xy)       if times_xy else np.array([])
        all_yz = np.concatenate(segs_yz, axis=0) if segs_yz else np.empty((0,2,2))
        all_ty = np.concatenate(times_yz)       if times_yz else np.array([])
        all_xz = np.concatenate(segs_xz, axis=0) if segs_xz else np.empty((0,2,2))
        all_tz = np.concatenate(times_xz)       if times_xz else np.array([])
        all_3d = np.concatenate(segs3d, axis=0) if segs3d else np.empty((0,2,3))
        all_t3 = np.concatenate(times3d)       if times3d else np.array([])

        # 2D plot
        fig, axs = plt.subplots(1,3, figsize=(18,6))
        for ax,segs,tarr,lims,title in [
            (axs[0], all_xy, all_tx, ('accX','accY'), f"{label} {which} {int(percent*100)}% 2D X-Y"),
            (axs[1], all_yz, all_ty, ('accY','accZ'), f"{label} {which} {int(percent*100)}% 2D Y-Z"),
            (axs[2], all_xz, all_tz, ('accX','accZ'), f"{label} {which} {int(percent*100)}% 2D X-Z")
        ]:
            lc = LineCollection(segs, cmap='viridis')
            lc.set_array(tarr); ax.add_collection(lc)
            # set limits
            arr = self.df[lims[0]].values;
            arr2 = self.df[lims[1]].values;
            ax.set_xlim(arr.min(), arr.max()); ax.set_ylim(arr2.min(), arr2.max())
            ax.set_title(title)
        fig.colorbar(LineCollection(all_xy, cmap='viridis'), ax=axs[0], label='Frame')
        plt.tight_layout()
        out2d = os.path.join(self.args.eval_plots_dir, f'{label}_{which}_2d_debug.pdf')
        fig.savefig(out2d); plt.close(fig)
        self.logger.info(f"Saved debug 2D plot: {out2d}")

        # 3D plot
        fig = plt.figure(figsize=(10,8))
        ax3 = fig.add_subplot(111, projection='3d')
        lc3 = Line3DCollection(all_3d, cmap='viridis')
        lc3.set_array(all_t3); ax3.add_collection(lc3)
        ax3.set_title(f"{label} {which} {int(percent*100)}% 3D")
        ax3.set_xlim(self.df['accX'].min(), self.df['accX'].max())
        ax3.set_ylim(self.df['accY'].min(), self.df['accY'].max())
        ax3.set_zlim(self.df['accZ'].min(), self.df['accZ'].max())
        fig.colorbar(lc3, ax=ax3, label='Frame')
        out3d = os.path.join(self.args.eval_plots_dir, f'{label}_{which}_3d_debug.pdf')
        fig.savefig(out3d); plt.close(fig)
        self.logger.info(f"Saved debug 3D plot: {out3d}")

    def run(self):
        self.load_or_train_model()
        self.evaluate()

        # Normal plots
        if not self.args.skip_normal_plots:
            # Use Plotter on new data; Plotter expects a behavior column
            eval_stats = {lbl: {'behavior': name} for lbl,name in self.behavior_map.items()}
            plotter = Plotter(self.args.output_csv, eval_stats, self.chunksize,
                              output_dir=self.args.eval_plots_dir)
            plotter.plot_overall()

        # Debug plots
        if self.args.debug_behaviors:
            # Group assigned distances by behavior
            by_beh = {}
            for s,e,l,d in self.assigned:
                by_beh.setdefault(l, []).append((s,e,d))
            for beh in self.args.debug_behaviors:
                lst = by_beh.get(beh, [])
                if not lst:
                    self.logger.warning(f"No chunks for behavior '{beh}' to debug.")
                    continue
                lst_sorted = sorted(lst, key=lambda x: x[2])
                top_n = max(1, int(len(lst_sorted)*self.args.debug_top_percent))
                bottom_n = max(1, int(len(lst_sorted)*self.args.debug_bottom_percent))
                self._plot_debug_chunks(lst_sorted[:top_n], beh, 'top', self.args.debug_top_percent)
                self._plot_debug_chunks(lst_sorted[-bottom_n:], beh, 'bottom', self.args.debug_bottom_percent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate new data with a pre-trained behavior classifier')
    parser.add_argument('-m', '--model-file', default='../classifier/training_stats.json',
                        help='path to classifier JSON file')
    parser.add_argument('-i', '--input', required=True,
                        help='path to new data CSV file')
    parser.add_argument('-c', '--chunksize', type=int, default=20,
                        help='number of samples per chunk')
    parser.add_argument('--train-input', default='../training_data/train_behavior.csv',
                        help='training CSV file if model needs retraining')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable debug logging')
    parser.add_argument('--skip-normal-plots', action='store_true', help='skip overall 2D/3D plots')
    parser.add_argument('--eval-plots-dir', default='eval_plots', help='directory to save evaluation plots')
    parser.add_argument('--debug-behaviors', nargs='+', choices=['s','t','l'], default=[],
                        help='behaviors to generate debug plots for')
    parser.add_argument('--debug-top-percent', type=float, default=0.1,
                        help='fraction of top confident chunks to debug-plot')
    parser.add_argument('--debug-bottom-percent', type=float, default=0.1,
                        help='fraction of least confident chunks to debug-plot')
    parser.add_argument('-o', '--output-csv', default='classified.csv',
                        help='path to save classified CSV')
    args = parser.parse_args()

    evaluator = Evaluator(
        model_file         = args.model_file,
        new_data_file      = args.input,
        training_input     = args.train_input,
        chunksize          = args.chunksize,
        output_csv         = args.output_csv,
        eval_plots_dir     = args.eval_plots_dir,
        verbose            = args.verbose,
        skip_normal_plots  = args.skip_normal_plots,
        debug_behaviors    = args.debug_behaviors,
        debug_top_percent  = args.debug_top_percent,
        debug_bottom_percent = args.debug_bottom_percent,
    )
    evaluator.run()

