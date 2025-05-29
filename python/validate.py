#!/usr/bin/env python3
"""
validate.py

Validate a trained snake behavior classifier against labeled data.

This script uses the shared Classifier logic to ensure consistency with evaluate.py.

Usage:
  python validate.py \
    --model-file classifier/training_stats.json \
    --labeled-data path/to/labeled.csv \
    [--chunksize 20]
"""
import argparse
import json
import sys
import logging

import pandas as pd
import numpy as np

from classifier import Classifier


def main():
    parser = argparse.ArgumentParser(description="Validate a snake classifier on labeled data")
    parser.add_argument('--model-file', '-m', default='classifier/training_stats.json',
                        help='path to classifier JSON file')
    parser.add_argument('--labeled-data', '-l', required=True,
                        help='path to labeled CSV (must include behavior column: accX,accY,accZ,behavior)')
    parser.add_argument('--chunksize', '-c', type=int, default=20,
                        help='number of samples per chunk')
    parser.add_argument('--supervised', action='store_true',
                        help='prompt on borderline chunks during validation')
    parser.add_argument('--borderline-threshold', type=float, default=0.1,
                        help='relative margin under which a chunk is considered borderline')
    args = parser.parse_args()

    # Set up logging to match evaluate.py format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('Validate')

    # Load model stats
    try:
        with open(args.model_file, 'r') as f:
            stats = json.load(f)
    except FileNotFoundError:
        logger.error(f"Model file not found: {args.model_file}")
        sys.exit(1)

    # Load labeled data
    try:
        df_labeled = pd.read_csv(args.labeled_data)
    except Exception as e:
        logger.error(f"Failed to read labeled data: {e}")
        sys.exit(1)

    if 'behavior' not in df_labeled.columns:
        logger.error("Labeled data must contain a 'behavior' column.")
        sys.exit(1)

    # Prepare unlabeled slice for classification
    df_unlabeled = df_labeled[['accX','accY','accZ']].copy()

    # Instantiate classifier and classify
    clf = Classifier(
        stats,
        args.chunksize,
        model_file=args.model_file,
        supervised=args.supervised,
        borderline_threshold=args.borderline_threshold
    )
    df_pred, _ = clf.classify(df_unlabeled)

    # Compare predictions to truth
    truth = df_labeled['behavior'].values
    preds = df_pred['predictedBehavior'].values
    if len(truth) != len(preds):
        logger.error("Prediction length does not match true labels.")
        sys.exit(1)

    accuracy = np.mean(preds == truth)
    logger.info(f"Validation accuracy: {accuracy*100:.2f}%")

    # Append accuracy to the model JSON
    try:
        with open(args.model_file, 'r+') as f:
            model_stats = json.load(f)
            model_stats['validation_accuracy'] = accuracy
            f.seek(0)
            json.dump(model_stats, f, indent=2)
            f.truncate()
        logger.info(f"Appended validation_accuracy to {args.model_file}")
    except Exception as e:
        logger.error(f"Failed to update model JSON: {e}")


if __name__ == '__main__':
    main()
