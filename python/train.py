#!/usr/bin/env python3
"""
train.py

This script defines a Trainer class that:
  - Loads a labeled training dataset (accX, accY, accZ, Behavior)
  - Computes per-chunk means and covariances (baseline: no outlier removal)
  - Saves behavior statistics to JSON and optionally invokes a plotter for visualization
"""
import argparse
import sys
import logging
import json
import os
import numpy as np
import pandas as pd

from utils import break_into_chunks
from plotter import Plotter

class Trainer:
    def __init__(self, training_file, chunksize=20, verbose=False):
        self.verbose = verbose
        self.training_file = training_file
        self.chunksize = chunksize
        self.setup_logging(verbose)
        self.logger = logging.getLogger(self.__class__.__name__)


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
            block = df_subset.iloc[start:end][['accX','accY','accZ']].values
            block_for_cov = block

            mean_vec = np.mean(block_for_cov, axis=0)
            if block_for_cov.shape[0] > 1:
                cov_mat = np.cov(block_for_cov, rowvar=False)
            else:
                cov_mat = np.zeros((3,3))
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
            df_sub = df[df['Behavior'] == label].copy()
            df_sub.sort_index(inplace=True)
            if df_sub.empty:
                self.logger.warning(f"No data for behavior '{label}' ({name}), skipping.")
                continue

            means, covs, ranges = self.compute_chunk_statistics(df_sub, self.chunksize, self.logger)
            if means:
                avg_mean = np.mean(means, axis=0).tolist()
                avg_cov = np.mean(covs, axis=0).tolist()
            else:
                avg_mean = [None, None, None]
                avg_cov = [[None]*3 for _ in range(3)]

            stats[label] = {
                'behavior': name,
                'average_mean': avg_mean,
                'average_covariance': avg_cov,
                # what are the chunk means and covariances doing here? Why isn't the average enough to get the information?
            }

            # pretty-print stats
            self.logger.info(f"Computed stats for '{name}':")
            self.logger.info(f"  Mean: {np.array(avg_mean)}")
            cov_arr = np.array(avg_cov)
            self.logger.info(f"  Covariance:\n{cov_arr}")

        return stats

    def save_statistics(self, stats):
        out_dir = "./classifier/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, 'training_stats.json')
        with open(out_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"Saved training statistics to {out_path}")
        return out_path

    def run(self):
        self.logger.info(f"Loading training data from {self.training_file}")
        stats = self.compute_training_statistics()
        stats_path = self.save_statistics(stats)

        # Optionally visualize results
        plotter = Plotter(self.training_file, stats, self.chunksize)
        plotter.plot_overall()

        return stats_path

    @staticmethod
    def setup_logging(verbose: bool):
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a snake behavior classifier by computing per-behavior statistics')
    parser.add_argument('-i', '--input', required=True, metavar='FILE', help='path to the training CSV file')
    parser.add_argument('-c', '--chunksize', type=int, default=20, help='the size of each behavior "chunk"')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable debug logging')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Error: input file not found: {args.input}")

    trainer = Trainer(
        training_file=args.input,
        chunksize=args.chunksize,
        verbose=args.verbose
    )
    trainer.run()

