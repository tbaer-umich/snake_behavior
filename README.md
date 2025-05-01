# Snake Behavior Classification

This project contains a simple python-based classifier and plotting scripts to categorize the behavior of timber rattlesnakes into either still, locomotion, or striking.

---

## Python Prerequisites
As a first step, you should have **Python 3.7+** installed on your machine. If you are using a Mac, open the Terminal application and type.
```bash
python --version
```
This should give you the currently installed python version.
You will also need some external packages to run this code, to install these dependencies running the following command in your terminal:
```bash
pip install numpy pandas matplotlib scipy
```
If you encounter errors installing these packages (you may see something like this), we will need to install a virtual enviroment.
```ansi
error: externally-managed-environment
× This environment is externally managed
└─> To install Python packages system-wide, try brew install <package>
```
Installing a virtual enviroment is thankfully straight forward, just type (or copy) the following into the terminal prompt:
```bash
python3 -m venv venv
source venv/bin/activate
```
Now you can rerun the `pip install` command to download all the external packages. Now we are ready to clone (download) this repo and start running code.

**Remember:** Every time you open up the terminal again to run this code, you will need to run the following command to re-launch the virtual enviroment.
```bash
source venv/bin/activate
```

---

## Installation
Installation is simple. To install this repository on to your local machine you will want to navigate (using the `cd` command) to a convienient location where you wish to download it. 

Here is an example of terminal caommands to navigate to your Desktop, run the `git clone` command which "checks out" a version of the github repository locally, and then enter the direcotry (named `snake_behavior`) using the `cd` command.
```bash
cd Desktop/
git clone git@github.com:tbaer-umich/snake_behavior.git
cd snake_behavior
```


## File Structure
Once downloaded, youn should find the following folder & files in the repository:
You can view these either by navigating to the folder in Finder or more convieniently by running the command `ls -l` in the terminal which shows everthing located in a folder/(directory)
```
├── train.py            # Trainer class & script
├── evaluate.py         # Evaluator class & script
├── plotter.py          # Plotter class & CLI
├── utils.py            # Helper: chunk-splitting function
├── training_data/      # Datasets used for training
├── data/               # Data we want to evaluate
├── categorized_data/   # Evaluated data
├── plots/              # Where .pdf outputs are stored by default
└── .gitignore          # A backend file to help setup git (may be invisible)
└── README.md           # This file
```

---

## Workflow Overview

So how do we use this classifier? There are three main files which each serve a different purpose throughout the workflow. In order, the workflow is:
1. **Train** a classifier on labeled data using `train.py`.
2. **Evaluate** new, unlabeled accelerometer data with `evaluate.py`.
3. **Visualize** results during training or evaluation via `plotter.py` (or the built‑in plot calls in the scripts).
Now let's go into some more detail on how to run each of these files.

---

## Trainer (`train.py`)

**Purpose:**
- Load a **labeled** CSV with columns `[accX, accY, accZ, Behavior]`.
- Split into fixed-size chunks (default 20 samples for now, but is variable).
- Compute each chunk’s mean & covariance, then average per behavior class. (This is what we use for classifying)
- Save statistics as JSON and generate initial plots.

**Internal class:** `Trainer`

**Arguments:**
To run script, we must call the python command `python train.py` with some additional arguments which give the script its inputs and/or options. The arguments are:
```bash
python train.py \
  -i ./path/to/train_data.csv   # required: your labeled CSV training data
  -o output_dir                 # optional: where to save JSON + plots (default `output`)
  -c 20                         # optional: chunk size in samples (default 20)
  -v                            # optional: verbose logging
```

**Example:**
If you are on the terminal inside of the `snake_behavior` folder, you should be able to run the training via this command:
```bash
python python/train.py -i ./training_data/labeled_data_0p8s.csv -o classifier -c 20 -v
```
After running, you’ll find:
- `./classifier/training_stats.json` (the extracted information from the training)
- Plots in `./plots/` (one 2D & 3D PDF per behavior)

---

## Evaluator (`evaluate.py`)

**Purpose:**
- Load a **trained** classifier JSON (or auto-train if missing). #TODO: test if auto-training works
- Load new **unlabeled** CSV with columns `[Date, Time, accX, accY, accZ]`.
- Split into chunks, compute covariance per chunk, measure AIRM distance to each behavior’s average covariance, assign best-match label.
- Save a new CSV with a `Behavior` column and optionally generate plots & debug visuals.

**Internal class:** `Evaluator`

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
python python/evaluate.py -i data/Eletra_1_test.csv \
                   -m classifier/training_stats.json \
                   -c 20 \
                   --eval-plots-dir plot/s \
                   -v
```
After running, you’ll find:
- `classified.csv` (or your chosen `-o` name) with a new `Behavior` column
- Plots in `plots/`:
  - `<prefix>_s_2d_cross.pdf`, `<prefix>_s_3d.pdf`, etc.
  - Debug plots for top/bottom confident chunks if requested

---

## Plotter (`plotter.py`)

**Purpose:**
- Standalone script to generate 2D cross-sections & 3D trajectories from **any** CSV that has a `Behavior` column. Usually plots are produced when running train or evaluate, but if needed they can also be created independantly.

**Usage:**
```bash
python plotter.py \
  -e categorized_data/Eletra_1_test_evaluated.csv  # CSV with `Behavior` column
  -c 20                     # chunk size
  -o plots/                 # output directory (default `plots`)
  -v                        # verbose logging
```

This produces `<prefix>_<behavior>_2d_cross.pdf` and `<prefix>_<behavior>_3d.pdf` in `plots/`.

---

## Utility (`utils.py`)

Contains some helper functions which are used during train, evaluate, and plot.
- **`break_into_chunks(df, chunk_size)`**: splits any table-like object into equal-length slices.

You generally don’t need to call it directly—it's used by `Trainer` and `Plotter` internally.

---

## Customization Tips

- Adjust **chunk size** (`-c`) depending on the chunk size expected in your training data
- Tweak **debug** percentages to get a closer look at the most/least confident predictions
- Extend `Evaluator` with post-processing rules (e.g. strike book-ending) as needed.

