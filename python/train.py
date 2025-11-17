#!/usr/bin/env python3
"""
train.py

This script defines a Trainer class that:
  - Loads a labeled training dataset (accX, accY, accZ, behavior)
  - Computes per-chunk means and covariances (baseline: no outlier removal)
  - Saves behavior statistics to JSON and optionally invokes a plotter for visualization
"""
import warnings
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

import argparse
import sys
import logging
import json
import os
import numpy as np
import pandas as pd

from utils import break_into_chunks, align_covariance_to_principal_axis
from plotter import Plotter

class Trainer:
    def __init__(self,
                 training_file: str,
                 chunksize: int = 20,
                 verbose: bool = False,
                 retrain: bool = False,
                 model_file: str = None):
        self.verbose = verbose
        self.training_file = training_file
        self.chunksize = chunksize
        self.verbose = verbose
        self.retrain = retrain
        self.model_file = model_file

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

        # If retraining, load existing model and validate chunksize
        self.existing_stats = None
        if self.retrain:
            if not self.model_file:
                raise ValueError("--retrain requires --model-file to be specified")
            with open(self.model_file, 'r') as f:
                self.existing_stats = json.load(f)
            # Extract chunksize from existing model if available
            if 'chunksize' in self.existing_stats and self.existing_stats['chunksize'] != self.chunksize:
                self.logger.warning(f"Using chunksize {self.existing_stats['chunksize']} from existing model (ignoring command line value {self.chunksize})")
                self.chunksize = self.existing_stats['chunksize']


    @staticmethod
    def compute_chunk_statistics(df_subset, chunksize, logger):
        """
        Given a DataFrame subset for one behavior, split into chunks
        and compute per-chunk mean and covariance (no outlier removal).
        #Returns lists: chunk_means, chunk_covs, chunk_ranges.
        """
        #logger.debug(f"The df subset is {df_subset.index}")
        chunks = break_into_chunks(df_subset, chunksize)

        means = []
        covs = []
        for start, end in chunks:
            if end <= start:
                continue
            block_df = df_subset.iloc[start:end]
            block = block_df[['accX','accY','accZ']].values

            # Special handling for strikes
            is_strike = block_df['behavior'].iloc[0] == 't'
            if is_strike and len(block) >= 25:
                # Compute per-sample variance
                per_sample_var = np.var(block, axis=1)
                center_idx = np.argmax(per_sample_var)
                half_window = 12  # 0.5s at 25Hz
                w_start = max(center_idx - half_window, 0)
                w_end = min(center_idx + half_window + 1, len(block))
                block_for_cov = block[w_start:w_end]
            else:
                block_for_cov = block

            mean_vec = np.mean(block_for_cov, axis=0)
            if block_for_cov.shape[0] > 1:
                cov_mat = np.cov(block_for_cov, rowvar=False)
            else:
                cov_mat = np.zeros((3,3))
            # Align covariance matrix to principal axis
            # Note: Mean vectors are kept in original reference frame
            # Only covariance matrices are rotated to align principal components
            #cov_mat = cov_mat = align_covariance_to_principal_axis(cov_mat) # NOTE: WIP, not yet reliable enough for use
            means.append(mean_vec.tolist())
            covs.append(cov_mat.tolist())

        return means, covs, chunks



    def compute_training_statistics(self):
        df = pd.read_csv(self.training_file)
        # I am not sure we really need this. Or if we do this it should be after we break everything into chunks
        # what part of the training statistics needs any information at all about the elapsed time???
        #df['time'] = np.arange(len(df)) / 25.0 # every 25 entries is a second

        behavior_map = {'s': 'Still', 't': 'Strike', 'l': 'Locomotion'}
        stats = {}

        for label, name in behavior_map.items():
            df_sub = df[df['behavior'] == label].copy()
            df_sub.sort_index(inplace=True)
            if df_sub.empty:
                self.logger.warning(f"No data for behavior '{label}' ({name}), skipping.")
                continue

            means, covs, ranges = self.compute_chunk_statistics(df_sub, self.chunksize, self.logger)
            if means: #this is a really janky if-guard TODO: think about removing this
                avg_mean = np.mean(means, axis=0).tolist()
                avg_cov = np.mean(covs, axis=0).tolist()
                num_chunks = len(means)
            else:
                avg_mean = [None, None, None]
                avg_cov = [[None]*3 for _ in range(3)]
                num_chunks = 0

            # If retraining, combine with existing statistics
            if self.retrain and self.existing_stats and label in self.existing_stats:
                old_stats = self.existing_stats[label]
                old_mean = np.array(old_stats['average_mean'])
                old_cov = np.array(old_stats['average_covariance'])
                old_num_chunks = old_stats.get('num_chunks', 0)

                if num_chunks > 0 and old_num_chunks > 0:
                    # Weighted average based on number of chunks
                    total_chunks = old_num_chunks + num_chunks
                    avg_mean = ((old_mean * old_num_chunks + np.array(avg_mean) * num_chunks) / total_chunks).tolist()
                    avg_cov = ((old_cov * old_num_chunks + np.array(avg_cov) * num_chunks) / total_chunks).tolist()
                    num_chunks = total_chunks
                    self.logger.info(f"Combined {old_num_chunks} existing chunks with {len(means)} new chunks for '{name}'")
                elif old_num_chunks > 0:
                    # No new chunks, keep old stats
                    avg_mean = old_mean.tolist()
                    avg_cov = old_cov.tolist()
                    num_chunks = old_num_chunks

            stats[label] = {
                'behavior': name,
                'average_mean': avg_mean,
                'average_covariance': avg_cov,
                'num_chunks': num_chunks,
            }

            # pretty-print stats
            self.logger.info(f"Computed stats for '{name}':")
            self.logger.info(f"  Mean: {np.array(avg_mean)}")
            cov_arr = np.array(avg_cov)
            self.logger.info(f"  Covariance:\n{cov_arr}")
            self.logger.info(f"  Number of chunks: {num_chunks}")

        # Add chunksize to stats for future retraining
        stats['chunksize'] = self.chunksize


        return stats

    def save_statistics(self, stats):
        # derive the “time period” (e.g. 0.8s or 5.0s) from the filename
        base = os.path.basename(self.training_file)
        name, _ = os.path.splitext(base)
        time_label = name.split('_')[-1]

        out_dir = "./classifier/"
        os.makedirs(out_dir, exist_ok=True)

        # If retraining, append _updated to the model filename
        if self.retrain:
            model_base = os.path.basename(self.model_file)
            model_name, _ = os.path.splitext(model_base)
            out_name = f"{model_name}_updated.json"
        else:
            out_name = f"training_stats_{time_label}.json"

        out_path = os.path.join(out_dir, out_name)

        with open(out_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"Saved training statistics to {out_path}")
        return out_path

    def run(self):
        self.logger.info(f"Loading training data from {self.training_file}")
        stats = self.compute_training_statistics()
        stats_path = self.save_statistics(stats)

        # Optionally visualize results
        if not self.retrain:
            plotter = Plotter(self.training_file, stats, self.chunksize)
            plotter.plot_overall()
        else:
            self.logger.info("Skipping plots during retraining")

        return stats_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a snake behavior classifier by computing per-behavior statistics')
    parser.add_argument('-i', '--input', required=True, metavar='FILE', help='path to the training CSV file')
    parser.add_argument('-c', '--chunksize', type=int, default=20, help='the size of each behavior "chunk"')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable debug logging')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain an existing model with additional data')
    parser.add_argument('-m', '--model-file', metavar='FILE', help='path to existing model JSON (required for --retrain)')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Error: input file not found: {args.input}")

    trainer = Trainer(
        training_file=args.input,
        chunksize=args.chunksize,
        verbose=args.verbose,
        retrain=args.retrain,
        model_file=args.model_file
    )
    trainer.run()

