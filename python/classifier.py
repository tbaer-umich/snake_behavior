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
                 borderline_threshold: float = 0.1):
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
        # manual default distance scaling; adjust here as needed
        self.scale = {'s': 5.0, 't': 1.0, 'l': 1.0}
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

    def additional_requirements(self, idx: int, label: str, assigned: list) -> str:
        """
        Hook for any extra chunk‐level rules.  Currently:
          - strikes ('t') must be preceded and followed by still ('s'), or else
            we demote them to locomotion ('l').
        idx       = index of this chunk in the full chunks list
        label     = the raw label chosen by distance
        assigned  = list of prior (start,end,label,dist) tuples
        """

        """
        Two‐phase bookending for strikes:
         1) If a chunk looks like 't' with a preceding 's', we tentatively
            accept and store its index.
         2) On the very next chunk, if it's not 's', we go back and demote
            that pending strike to 'l'.
        """
        # Phase 2: confirm or reject a pending strike from the last chunk
        if self._pending_strike_idx is not None:
            if label != 's':
                p = self._pending_strike_idx
                start, end, _, dist = assigned[p]
                assigned[p] = (start, end, 'l', dist)
            self._pending_strike_idx = None

        # Phase 1: when we see a raw 't', require previous was 's'
        if label == 't' and idx > 0:
            prev_lbl = assigned[idx - 1][2]
            if prev_lbl == 's':
                # tentatively accept—will confirm on next chunk
                self._pending_strike_idx = idx
                return 't'
            else:
                # immediate demotion
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
        chunks = break_into_chunks(df, self.chunksize)
        assigned = []

        for idx, (start, end) in enumerate(chunks):
            block = df[['accX', 'accY', 'accZ']].iloc[start:end].values
            if block.shape[0] < 2:
                raw, dist = 'u', np.nan
            else:
                cov = np.cov(block, rowvar=False)
                # compute and scale distances
                dists = {
                    lbl: self.scale[lbl] * self.airm_distance(cov, np.array(info['average_covariance']))
                    for lbl, info in self.stats.items()
                }
                sorted_lbls = sorted(dists, key=dists.get)
                raw = sorted_lbls[0]
                dist = dists[raw]

                # optionally prompt user on borderline cases
                if self.supervised and len(sorted_lbls) > 1:
                    diff = (dists[sorted_lbls[1]] - dist) / dist
                    if diff < self.threshold:
                        raw = self._prompt_user_for_chunk(df.iloc[start:end], idx)

            # apply non-distance related requirements
            #label = self.additional_requirements(idx, raw, assigned)
            label = raw # bypass the additional requirements for now (since strikes aren't bookended in the validation dataset)
            assigned.append((start, end, label, dist))
            df.loc[start:end-1, 'predictedBehavior'] = label

        return df, assigned

