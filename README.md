# Snake Behavior Classification

This repository provides a simple Python-based pipeline to **train** a basic behavior classifier from labeled accelerometer data, **evaluate** new unlabeled data, and **visualize** results as plots.

---

## File Structure

```
├── train.py            # Trainer class & script
├── evaluate.py         # Evaluator class & script
├── plotter.py          # Plotter class & CLI
├── utils.py            # Helper: chunk-splitting function
├── training_data/      # Datasets used for training
├── data/               # Data we want to evaluate
├── categorized_data/   # This is where our data goes after we evaluate it
├── plots/              # Where .pdf outputs are stored by default
└── README.md           # This file
```

---

## Prerequisites

- **Python 3.7+** installed
- Basic packages (install via `pip`):
  ```bash
  pip install numpy pandas matplotlib scipy
  ```

---

## Installation

1. Clone this repo:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scipy
   ```
3. (Optional) If you are having trouble installing the dependancies, try creating a virtual enviroment first
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

---

## Workflow Overview

1. **Train** a classifier on labeled data using `train.py`.
2. **Evaluate** new, unlabeled accelerometer data with `evaluate.py`.
3. **Visualize** results during training or evaluation via `plotter.py` (or the built‑in plot calls in the scripts).

---

## Trainer (`train.py`)

**Purpose:**
- Load a **labeled** CSV with columns `[accX, accY, accZ, Behavior]`.
- Split into fixed-size chunks (default 20 samples for now, but is variable).
- Compute each chunk’s mean & covariance, then average per behavior class. (This is what we use for classifying)
- Save statistics as JSON and generate initial plots.

**Key class:** `Trainer`

**Arguments:**
```bash
python train.py \
  -i ./path/to/train_data.csv    # required: your labeled CSV training data
  -o output_dir                 # optional: where to save JSON + plots (default `output`)
  -c 20                         # optional: chunk size in samples (default 20)
  -v                            # optional: verbose logging
```

**Example:**
```bash
python train.py -i ./training_data/labeled_data_0p8s.csv -o classifier -c 20 -v
```
After running, you’ll find:
- `./classifier/training_stats.json`
- Plots in `./plots/` (one 2D & 3D PDF per behavior)

---

## Evaluator (`evaluate.py`)

**Purpose:**
- Load a **trained** classifier JSON (or auto-train if missing). #TODO: test if this works
- Load new **unlabeled** CSV with columns `[Date, Time, accX, accY, accZ]`.
- Split into chunks, compute covariance per chunk, measure AIRM distance to each behavior’s average covariance, assign best-match label.
- Save a new CSV with a `Behavior` column and optionally generate plots & debug visuals.

**Key class:** `Evaluator`

**Arguments:**
```bash
python evaluate.py \
  -i data/new_data.csv              # required: new unlabeled data
  -m classifier/training_stats.json # optional: path to JSON (auto-trains if missing)
  --train-input train_behavior.csv  # optional: training CSV if retraining needed
  -c 20                             # chunk size (samples)
  -o categorized/data/              # where to save labeled output (default is classified.csv)
  --eval-plots-dir eval_plots       # directory for evaluation plots
  --skip-normal-plots               # skip overall 2D/3D plots
  --debug-behaviors s t             # optional: e.g. ['s','t'] to debug specific classes
  --debug-top-percent 0.1           # optional: fraction of top-confidence chunks to plot
  --debug-bottom-percent 0.1        # optional: fraction of low-confidence chunks to plot
  -v                                # verbose logging
```

**Example:**
```bash
python evaluate.py -i data/Eletra_1_test.csv \
                   -m classifier/training_stats.json \
                   -c 20 \
                   --eval-plots-dir plot/s \
                   -v
```
After running, you’ll find:
- `classified.csv` (or your chosen `-o`) with a new `Behavior` column
- Plots in `plots/`:
  - `<prefix>_s_2d_cross.pdf`, `<prefix>_s_3d.pdf`, etc.
  - Debug plots for top/bottom confident chunks if requested

---

## Plotter (`plotter.py`)

**Purpose:**
- Standalone script to generate 2D cross-sections & 3D trajectories from **any** CSV that has a `Behavior` column.

**Usage:**
```bash
python plotter.py \
  -e categorized_data/Eletra_1_test_evaluated.csv  # CSV with `Behavior` column
  -c 20                     # chunk size
  -o plots/                 # output directory (default `plots`)
  -v                        # verbose logging
```

This produces `<prefix>_<behavior>_2d_cross.pdf` and `<prefix>_<behavior>_3d.pdf` in `myplots/`.

---

## Utility (`utils.py`)

- **`break_into_chunks(df, chunk_size)`**: splits any table-like object into equal-length slices.

You generally don’t need to call it directly—it's used by `Trainer` and `Plotter` internally.

---

## Customization Tips

- Adjust **chunk size** (`-c`) depending on the chunk size expected in your training data
- Tweak **debug** percentages to get a closer look at the most/least confident predictions
- Extend `Evaluator` with post-processing rules (e.g. strike book-ending) as needed.

