#!/usr/bin/env python3
"""
validate.py

Validate a trained snake behavior classifier against labeled data and produce
publication-quality evaluation outputs.

Run from snake_behavior/ (the repo root).

Usage:
  python python/validate.py \\
    --labeled-data validation_data/doja_validation_data.csv \\
    [--model-file classifier/training_stats.json] \\
    [--output-dir evaluation] \\
    [--chunksize 20]

Outputs (written to --output-dir):
  confusion_matrix.csv       Raw counts confusion matrix
  metrics.csv                Per-class Accuracy, Precision, Recall, F-score
  confusion_matrix_fig.pdf   Publication-quality heatmap (viridis, annotated)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from classifier import Classifier

# ── Canonical class order for axes (matches AcceleRater paper convention) ──────
CLASS_ORDER = ['stillness', 'locomotion', 'strikes']
# Short tick labels for the figure axes  (s / l / t)
CLASS_LABELS = {'stillness': 's', 'locomotion': 'l', 'strikes': 't'}


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(truth: np.ndarray, preds: np.ndarray, classes: list) -> pd.DataFrame:
    """
    Return a DataFrame with per-class Accuracy, Precision, Recall, and F-score.

    All four metrics are computed from the binary TP/TN/FP/FN counts for each
    class treated as a one-vs-rest problem, matching the AcceleRater convention.
    """
    rows = []
    n = len(truth)
    for cls in classes:
        tp = np.sum((truth == cls) & (preds == cls))
        tn = np.sum((truth != cls) & (preds != cls))
        fp = np.sum((truth != cls) & (preds == cls))
        fn = np.sum((truth == cls) & (preds != cls))

        accuracy  = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score   = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        rows.append({
            'behavior': cls,
            'accuracy':  round(accuracy,  4),
            'precision': round(precision, 4),
            'recall':    round(recall,    4),
            'f_score':   round(f_score,   4),
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        })
    return pd.DataFrame(rows)


def compute_confusion_matrix(truth: np.ndarray, preds: np.ndarray,
                              classes: list) -> pd.DataFrame:
    """
    Return a DataFrame with raw counts.  Rows = true label, columns = predicted.
    """
    cm = pd.DataFrame(0, index=classes, columns=classes)
    for t, p in zip(truth, preds):
        if t in cm.index and p in cm.columns:
            cm.loc[t, p] += 1
    return cm


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_counts: pd.DataFrame, out_path: Path) -> None:
    """
    Publication-quality confusion matrix heatmap.

    Style mirrors the AcceleRater paper figures:
      - viridis colormap, row-normalised (0–100 %)
      - each cell annotated with both the raw count and the row-percentage
      - single-letter tick labels (s / l / t)
      - clean white grid, no top/right spines
    """
    classes = list(cm_counts.index)
    n = len(classes)

    counts = cm_counts.values.astype(float)
    row_sums = counts.sum(axis=1, keepdims=True)
    # avoid division by zero for empty rows
    normed = np.where(row_sums > 0, counts / row_sums * 100, 0.0)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(normed, cmap='viridis', vmin=0, vmax=100, aspect='equal')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Cell annotations: percentage on top, raw count below in smaller text
    for i in range(n):
        for j in range(n):
            pct = normed[i, j]
            cnt = int(counts[i, j])
            # choose white or black text depending on cell brightness
            text_color = 'white' if pct < 60 else 'black'
            ax.text(j, i - 0.12, f'{pct:.1f}%',
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color=text_color)
            ax.text(j, i + 0.22, f'n={cnt}',
                    ha='center', va='center', fontsize=7,
                    color=text_color, alpha=0.85)

    # Tick labels (single letter, matching AcceleRater convention)
    tick_labels = [CLASS_LABELS.get(c, c) for c in classes]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=10)

    ax.set_xlabel('Predicted', fontsize=10, labelpad=6)
    ax.set_ylabel('True', fontsize=10, labelpad=6)

    # Light grid between cells
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Remove top/right spines for a clean journal look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate a snake behavior classifier and produce publication outputs.')
    parser.add_argument('--model-file', '-m',
                        default='classifier/training_stats.json',
                        help='path to classifier JSON file (default: classifier/training_stats.json)')
    parser.add_argument('--labeled-data', '-l', required=True,
                        help='path to labeled CSV with columns: accX, accY, accZ, behavior')
    parser.add_argument('--output-dir', '-o',
                        default='evaluation',
                        help='directory for output files (default: evaluation/)')
    parser.add_argument('--chunksize', '-c', type=int, default=20,
                        help='number of samples per chunk (default: 20)')
    parser.add_argument('--supervised', action='store_true',
                        help='prompt on borderline chunks during validation')
    parser.add_argument('--borderline-threshold', type=float, default=0.1,
                        help='relative margin below which a chunk is borderline (default: 0.1)')
    parser.add_argument('--debug-strikes', action='store_true',
                        help='generate debug plots for all chunks classified as strikes')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('Validate')

    # ── Output directory ───────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        with open(args.model_file, 'r') as f:
            stats = json.load(f)
    except FileNotFoundError:
        logger.error(f'Model file not found: {args.model_file}')
        sys.exit(1)

    # ── Load labeled data ──────────────────────────────────────────────────────
    labeled_path = Path(args.labeled_data)
    if not labeled_path.exists():
        logger.error(f'Labeled data file not found: {labeled_path}')
        sys.exit(1)

    logger.info(f'Loading labeled data: {labeled_path}')
    try:
        df_labeled = pd.read_csv(labeled_path)
    except Exception as e:
        logger.error(f'Failed to read labeled data: {e}')
        sys.exit(1)

    if 'behavior' not in df_labeled.columns:
        logger.error("Labeled data must contain a 'behavior' column.")
        sys.exit(1)

    logger.info(f'Loaded {len(df_labeled):,} rows from {labeled_path.name}')

    # ── Run classifier ─────────────────────────────────────────────────────────
    df_unlabeled = df_labeled[['accX', 'accY', 'accZ']].copy()

    clf = Classifier(
        stats,
        args.chunksize,
        model_file=args.model_file,
        supervised=args.supervised,
        borderline_threshold=args.borderline_threshold,
        debug_strikes=args.debug_strikes,
    )
    df_pred, _ = clf.classify(df_unlabeled)

    # ── Align predictions with ground truth ────────────────────────────────────
    truth = df_labeled['behavior'].values
    preds = df_pred['predictedBehavior'].values

    valid_mask = pd.notna(truth)
    n_skipped = int((~valid_mask).sum())
    if n_skipped:
        logger.warning(f'Skipping {n_skipped:,} rows with missing labels')
    truth = truth[valid_mask]
    preds = preds[valid_mask]

    if len(truth) != len(preds):
        logger.error('Prediction length does not match true label length after filtering.')
        sys.exit(1)

    logger.info(f'Evaluating {len(truth):,} labeled chunks')

    # ── Determine class order ──────────────────────────────────────────────────
    # Use canonical order where possible; append any unexpected classes found in data
    present = list(np.unique(truth))
    classes = [c for c in CLASS_ORDER if c in present] + \
              [c for c in present if c not in CLASS_ORDER]
    if set(classes) != set(present):
        logger.warning(f'Unexpected classes in data: {set(present) - set(CLASS_ORDER)}')

    # ── Overall accuracy ───────────────────────────────────────────────────────
    overall_acc = float(np.mean(preds == truth))
    logger.info(f'Overall accuracy: {overall_acc * 100:.2f}%')

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = compute_confusion_matrix(truth, preds, classes)
    cm_path = out_dir / 'confusion_matrix.csv'
    cm.to_csv(cm_path)
    logger.info(f'Confusion matrix saved → {cm_path}')

    # ── Per-class metrics ──────────────────────────────────────────────────────
    metrics_df = compute_metrics(truth, preds, classes)
    metrics_path = out_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f'Metrics saved → {metrics_path}')

    # Print a readable summary to the console
    logger.info('─' * 60)
    logger.info(f'{"Behavior":<14} {"Accuracy":>9} {"Precision":>10} {"Recall":>8} {"F-score":>8}')
    logger.info('─' * 60)
    for _, row in metrics_df.iterrows():
        logger.info(
            f'{row["behavior"]:<14} '
            f'{row["accuracy"]*100:>8.2f}% '
            f'{row["precision"]*100:>9.2f}% '
            f'{row["recall"]*100:>7.2f}% '
            f'{row["f_score"]*100:>7.2f}%'
        )
    logger.info('─' * 60)

    # Misclassification detail (kept from original script)
    for cls in classes:
        idx = truth == cls
        total = int(idx.sum())
        if total == 0:
            continue
        correct = int(np.sum(preds[idx] == cls))
        mis = total - correct
        logger.info(
            f"Behavior '{cls}': {correct}/{total} correct, "
            f"{mis}/{total} misclassified ({mis/total*100:.2f}%)"
        )
        for other in classes:
            if other == cls:
                continue
            count = int(np.sum((truth == cls) & (preds == other)))
            if count > 0:
                logger.info(
                    f"    → misclassified as '{other}': {count} ({count/total*100:.2f}%)"
                )

    # ── Confusion matrix figure ────────────────────────────────────────────────
    fig_path = out_dir / 'confusion_matrix_fig.pdf'
    plot_confusion_matrix(cm, fig_path)
    logger.info(f'Figure saved → {fig_path}')

    # ── Update model JSON with validation results ──────────────────────────────
    try:
        with open(args.model_file, 'r+') as f:
            model_stats = json.load(f)
            model_stats['validation_accuracy'] = overall_acc
            model_stats['validation_metrics'] = metrics_df[
                ['behavior', 'accuracy', 'precision', 'recall', 'f_score']
            ].to_dict(orient='records')
            f.seek(0)
            json.dump(model_stats, f, indent=2)
            f.truncate()
        logger.info(f'Updated validation results in {args.model_file}')
    except Exception as e:
        logger.error(f'Failed to update model JSON: {e}')


if __name__ == '__main__':
    main()
