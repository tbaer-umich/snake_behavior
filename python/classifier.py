#!/usr/bin/env python3
"""
classifier.py

Shared classification logic for snake behavior:
  - Computes AIRM distance
  - Applies manual distance scaling
  - Enforces strike bookending
  - Optionally prompts user on borderline chunks
  - Classifies a DataFrame of accX, accY, accZ samples into behavior labels
"""
import warnings
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm, logm, norm
from plotter import Plotter
from utils import break_into_chunks


class Classifier:
    def __init__(self,
                 stats: dict,
                 chunksize: int,
                 model_file: str = None,
                 supervised: bool = False,
                 borderline_threshold: float = 0.1,
                 debug_strikes: bool = False):
        """
        stats: loaded JSON stats mapping label -> dict with 'average_covariance'
        chunksize: number of samples per chunk
        model_file: optional path to JSON for persistent updates
        supervised: if True, prompt user on borderline cases
        borderline_threshold: relative margin to consider a chunk borderline
        """
        # filter out any entries that aren’t valid behavior stats
        self.stats = {
            k: v for k, v in stats.items()
            if isinstance(v, dict) and 'average_covariance' in v
        }
        self.chunksize = chunksize
        self.model_file = model_file
        self.supervised = supervised
        self.threshold = borderline_threshold
        self.debug_strikes = debug_strikes
        # manual default distance scaling; adjust here as needed
        self.scale = {'s': 0.75, 't': 0.5, 'l': 1.0}
        self.sampling_rate = 25  # samples per second
        # for two-phase strike bookending
        self._pending_strike_idx = None

    def airm_distance(self, cov1, cov2, eps=1e-8) -> float:
        cov1_reg = cov1 + eps * np.eye(cov1.shape[0])
        cov2_reg = cov2 + eps * np.eye(cov2.shape[0])
        sqrt_cov1 = sqrtm(cov1_reg)
        inv_sqrt = np.linalg.inv(sqrt_cov1)
        inner = inv_sqrt @ cov2_reg @ inv_sqrt
        log_inner = logm(inner)
        return norm(log_inner, 'fro')

    def bookended_by_stillness_chunks(self, idx: int, assigned: list) -> bool:
        """
        Original two-phase bookending:
         Phase 2: confirm or reject any pending strike
         Phase 1: if this chunk is a raw 't', require previous was 's'
        """
        # need at least one chunk before and after
        if idx <= 0 or idx >= len(assigned) - 1:
            return False
        prev_lbl = assigned[idx - 1][2]
        next_lbl = assigned[idx + 1][2]
        return prev_lbl == 's' and next_lbl == 's'
        # Phase 2: resolve pending strike from prior chunk
        if self._pending_strike_idx is not None:
            # if this chunk isn't still, demote the pending strike
            if assigned[idx][2] != 's':
                p = self._pending_strike_idx
                start, end, _, dist = assigned[p]
                assigned[p] = (start, end, 'l', dist)
            self._pending_strike_idx = None

        # Phase 1: when we see a raw 't', check previous was 's'
        raw_label = assigned[idx][2]
        if raw_label == 't' and idx > 0:
            prev_lbl = assigned[idx - 1][2]
            if prev_lbl == 's':
                # tentatively accept; will confirm on next chunk
                self._pending_strike_idx = idx
                return True
            else:
                # immediate demotion
                return False

        # non-strikes or default pass
        return True

    def bookended_by_stillness_neighborhood(self, idx: int, assigned: list) -> bool:
        """
        New bookending logic: check stillness within ±2.5s around the 1s strike window.
        """
        # ensure valid strike candidate
        start, end, label, _ = assigned[idx]
        if label != 't':
            return False
        # find center of strike: highest variance sample
        block = self.df_all[['accX', 'accY', 'accZ']].iloc[start:end]
        variances = block.var(axis=1)
        center_i = variances.idxmax()
        # define windows in samples
        radius = int(2.5 * self.sampling_rate)  # 2.5s neighborhood
        strike_win = int(1.0 * self.sampling_rate)  # 1s strike
        half_strike = strike_win // 2
        # compute slice bounds, clamp to data range
        before_start = max(0, center_i - radius)
        before_end   = max(before_start, center_i - half_strike)
        after_start  = min(len(self.df_all), center_i + half_strike)
        after_end    = min(len(self.df_all), center_i + radius)

        def region_is_still(i_start: int, i_end: int) -> bool:
            if i_end <= i_start:
                return False
            df_region = self.df_all[['accX', 'accY', 'accZ']].iloc[i_start:i_end]
            cov = 0.75*np.cov(df_region.values, rowvar=False) # manually editing this region_is_still distance
            dists = {
                lbl: self.scale[lbl] * self.airm_distance(
                    cov, np.array(info['average_covariance'])
                )
                for lbl, info in self.stats.items()
            }
            pred = min(dists, key=dists.get)
            return pred == 's'

        # require both pre- and post-strike regions to be still
        if not region_is_still(before_start, before_end):
            return False
        if not region_is_still(after_start, after_end):
            return False
        return True

    def additional_requirements(self, idx: int, label: str, assigned: list) -> str:
        """
        Hook for extra chunk‐level rules. Now supports two bookending approaches:
          - chunk-based (bookended_by_stillness_chunks)
          - neighborhood-based (bookended_by_stillness_neighborhood)

        We preserve the old two-phase code in comments for reference.
        """
        # Approach 1: original two-phase chunk bookending
        #if label == 't' and not self.bookended_by_stillness_chunks(idx, assigned):
        #    return 'l'

        # Approach 2: neighborhood-based method (comment out to compare)
        if label == 't' and not self.bookended_by_stillness_neighborhood(idx, assigned):
            return 'l'

        return label

    def _prompt_user_for_chunk(self, chunk_df, idx: int) -> str:
        """
        Use Plotter to display time-series for this chunk,
        then prompt user to choose a label.
        """
        # instantiate minimal Plotter with required attributes
        plotter = Plotter.__new__(Plotter)
        plotter.df = chunk_df.reset_index(drop=True)
        plotter.prefix = f"chunk_{idx}"
        plotter.output_dir = "."
        # show interactively (blocks until window is closed)
        plotter.plot_time_series(plotter.df, label=f"{idx}", interactive=True)

        valid = set(self.stats.keys())
        choice = None
        while choice not in valid:
            choice = input(f"Enter label for chunk {idx} {valid}: ").strip()
        return choice

    def classify(self, df: np.ndarray) -> tuple:
        """
        Break df into chunks, compute covariance distances, apply scaling,
        enforce bookending, optionally prompt on borderline, and assign
        'predictedBehavior'.
        Returns (new_df, assigned_list).
        assigned_list: list of tuples (start, end, label, distance).
        """
        df = df.copy()
        df['predictedBehavior'] = ''
        # keep full trace for neighborhood logic
        self.df_all = df
        chunks = break_into_chunks(df, self.chunksize)
        assigned = []

        for idx, (start, end) in enumerate(chunks):
            block = df[['accX', 'accY', 'accZ']].iloc[start:end].values
            if block.shape[0] < 2:
                raw, dist = 'u', np.nan
            else:
                dists = {}
                for lbl, info in self.stats.items():
                    if lbl == 't':
                        # Use 1s center window for strikes
                        per_sample_var = np.var(block, axis=1)
                        center_idx = np.argmax(per_sample_var)
                        half = self.sampling_rate // 2  # 0.5s
                        w_start = max(center_idx - half, 0)
                        w_end = min(center_idx + half + 1, block.shape[0])
                        strike_block = block[w_start:w_end]
                        if strike_block.shape[0] >= 2:
                            cov = np.cov(strike_block, rowvar=False)
                        else:
                            cov = np.eye(3)
                    else:
                        cov = np.cov(block, rowvar=False)
                    dists[lbl] = self.scale[lbl] * self.airm_distance(
                        cov, np.array(info['average_covariance'])
                    )
                sorted_lbls = sorted(dists, key=dists.get)
                raw = sorted_lbls[0]
                dist = dists[raw]

                # optionally prompt user on borderline cases
                if self.supervised and len(sorted_lbls) > 1:
                    diff = (dists[sorted_lbls[1]] - dist) / dist
                    if diff < self.threshold:
                        raw = self._prompt_user_for_chunk(df.iloc[start:end], idx)

            # apply non-distance related requirements
            temp_assigned = assigned + [(start, end, raw, dist)]
            label = self.additional_requirements(idx, raw, temp_assigned)
            #label = raw # bypass the additional requirements for now (since strikes aren't bookended in the validation dataset)
            assigned.append((start, end, label, dist))

            if self.debug_strikes and label == 't':
                block_df = df.iloc[start:end].copy()
                block = block_df[['accX', 'accY', 'accZ']].values
                var = np.var(block, axis=1)
                center_idx = np.argmax(var)

                sr = self.sampling_rate
                half_strike = sr // 2
                full_neigh = int(2.5 * sr)

                center_global = start + center_idx
                strike_start = max(center_global - half_strike, 0)
                strike_end = min(center_global + half_strike + 1, len(df))
                neigh_start = max(center_global - full_neigh, 0)
                neigh_end = min(center_global + full_neigh + 1, len(df))

                debug_df = df[['accX', 'accY', 'accZ']].iloc[neigh_start:neigh_end].copy()
                debug_df = debug_df.reset_index(drop=True)

                plotter = Plotter.__new__(Plotter)
                plotter.df = debug_df
                plotter.prefix = f"strike_debug_{idx}"
                plotter.output_dir = "./debug_strikes"
                plotter.plot_strike_debug(
                    debug_df,
                    strike_start - neigh_start,
                    strike_end - neigh_start,
                    center_idx=center_global - neigh_start
                )
            df.loc[start:end-1, 'predictedBehavior'] = label

        return df, assigned

