# Snake Behavior Classification

This project contains a simple python-based classifier and plotting scripts to categorize the behavior of timber rattlesnakes into either still, locomotion, or striking.

For absolute beginners to python --> start with the [beginner's guide](#appendix-beginners-guide) at the bottom of this file.

---

## Quick Setup (Mac/Linux)

### Automatic Setup (Recommended)
We've created a setup script that handles everything for you:
```bash
# 1. Clone the repository
git clone https://github.com/tbaer-umich/snake_behavior.git
cd snake_behavior

# 2. Run the setup script
./setup.sh
```

The setup script will:
- Check that Python 3.7+ is installed
- Create a virtual environment
- Install all required packages
- Check for GUI support (Tkinter)

### Manual Setup (If Automatic Fails)
If the automatic setup doesn't work, you can set up manually:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Note for Mac Users:** If you get an error about Tkinter when running the labeler, you may need to:
- Either: Install Python from [python.org](https://www.python.org/downloads/) (includes Tkinter)
- Or: Install via Homebrew: `brew install python-tk`

---

## Updating the Code

To get the latest updates and improvements:
```bash
# Navigate to the project folder
cd snake_behavior

# Get and apply updates from GitHub
git fetch origin
git pull
```

---

## Getting Started

After setup is complete, activate your environment and start labeling:
```bash
# Always activate the environment first (every time you open terminal)
source venv/bin/activate

# Start the labeling tool
python python/labeler.py
```

---

## Main Tools

### 1. [Labeler](#labeler-labelerpy) (`labeler.py`) - Start Here!
**What it does:** Interactive GUI for labeling snake accelerometer data

**How to use:**
```bash
python python/labeler.py
```
- Click "Load Data" to open a CSV file with accelerometer data
- View 5-second chunks and classify as: Still (1), Locomotion (2), Strike (3), or Uncertain (4)
- Use keyboard shortcuts: A (previous), D (next), 1-4 (classify)
- Auto-saves every 25 labels to `*_labeled.csv`

**Pro tip:** Load a trained classifier to get AI predictions - just press D to accept them!

### 2. [Trainer](#trainer-trainpy) (`train.py`)
**What it does:** Trains a classifier from labeled data

**How to use:**
```bash
python python/train.py -i training_data/your_labeled_data.csv -c 125
```
Creates `classifier/training_stats.json` containing the trained model

### 3. [Evaluator](#evaluator-evaluatepy) (`evaluate.py`)
**What it does:** Classifies new unlabeled data using a trained model

**How to use:**
```bash
python python/evaluate.py -i data/new_data.csv -m classifier/training_stats.json -c 125
```
Creates `classified.csv` with behavior predictions

### 4. [Validator](#validator-validatepy) (`validate.py`)
**What it does:** Tests classifier accuracy on labeled data

**How to use:**
```bash
python python/validate.py -m classifier/training_stats.json -l training_data/test_data.csv -c 125
```

---

## File Structure
Once downloaded, youn should find the following folder & files in the repository:
You can view these either by navigating to the folder in Finder or more convieniently by running the command `ls -l` in the terminal which shows everthing located in a folder/(directory)
```
├── python/               # All main Python scripts
│   ├── train.py          # Trainer class & script
│   ├── evaluate.py       # Evaluator class & script
│   ├── classifier.py     # Shared classification logic (used by evaluate & validate)
│   ├── validate.py       # Validator class & script for labeled data
│   ├── plotter.py        # Plotter class & CLI
│   ├── labeler.py        # GUI tool for manually labeling accelerometer data
│   └── utils.py          # Helper: chunk-splitting function
├── classifier/           # Generated JSON models (e.g. training_stats_*.json)
├── training_data/        # Datasets used for training
├── data/                 # Data we want to evaluate
├── categorized_data/     # Evaluated data (CSV outputs)
├── plots/                # Where .pdf outputs are stored by default
├─── .gitignore           # Excluded files
└── README.md             # This file
```

---

## Workflow Overview

So how do we use this classifier? There are four main files which each serve a different purpose throughout the workflow. In order, the workflow is:
1. **Train** a classifier on labeled data using `train.py`.
2. **Evaluate** new, unlabeled accelerometer data with `evaluate.py`.
3. **Validate** classifier performance against held-out labeled data using `validate.py`.
4. **Visualize** results during training or evaluation via `plotter.py` (or the built-in plot calls in the scripts).
Now let's go into some more detail on how to run each of these files.

---

## Labeler (`labeler.py`)

**Purpose:**
- Provides a graphical user interface (GUI) for manually labeling accelerometer data from timber rattlesnakes
- Displays 5-second chunks (125 samples at 25Hz) of accelerometer data as time-series plots
- Allows classification into four categories: Still, Locomotion, Strike, or Uncertain
- Supports resuming partially-labeled datasets to continue where you left off
- Can load pre-trained classifier predictions to speed up the labeling process through semi-automated review

**How to Run:**
```bash
python python/labeler.py
```
This will open a GUI window. No command-line arguments are needed - all operations are done through the interface.

**Basic Usage:**
1. **Starting Fresh:**
   - Click "Load Data" to select an unlabeled CSV file with columns: `Date, Time, accX, accY, accZ`
   - The tool will display the first 5-second chunk of data as three time-series plots (one for each axis)
   - Click one of the four behavior buttons or press keyboard shortcuts (1-4) to classify the current chunk
   - Use navigation buttons or keyboard shortcuts to move between chunks:
     - Press 'A' or '← Previous' button to go back
     - Press 'D', Space, or 'Next →' button to advance
   - Classifications auto-save every 25 labels to a file named `<original_filename>_labeled.csv`

2. **Resuming Previous Work:**
   - Click "Load Progress" to select a `*_labeled.csv` file you were previously working on
   - The tool automatically loads your previous labels and jumps to the first unlabeled chunk
   - Continue labeling from where you left off

3. **Using Classifier Predictions (Semi-Automated Mode):**
   - First load your data using either "Load Data" or "Load Progress"
   - Click "Get Classifier Predictions" and select a trained classifier JSON file (usually in `./classifier/`)
   - The classifier will run on the entire dataset (this may take a moment)
   - As you navigate chunks, classifier predictions appear in orange with "(classifier prediction)" suffix
   - To accept a prediction: Simply press 'D' to move to the next chunk - this auto-confirms the prediction
   - To override a prediction: Click any classification button - it will turn green and show "(revised)" if different from the prediction
   - All confirmed predictions and manual labels are saved when you save the file

**Keyboard Shortcuts:**
- `1` - Classify as Still
- `2` - Classify as Locomotion
- `3` - Classify as Strike
- `4` - Classify as Uncertain
- `A` - Previous chunk
- `D` or `Space` - Next chunk (auto-confirms classifier predictions if present)
- `+` - Zoom out (increase y-axis range)
- `-` - Zoom in (decrease y-axis range)

**Additional Features:**
- **Zoom In/Out buttons:** Adjust the vertical scale of plots to see more or less detail. The zoom level persists as you navigate through chunks.
- **Show Context button:** Opens a popup window showing 5 chunks before and after the current chunk, with all three axes plotted together in different colors.
- **Go to chunk:** Type a chunk number in the text box and click "Go" to jump directly to that chunk without cycling through all chunks.
- **Show Strikes Only button:** (Available after loading classifier predictions) Toggle to navigate only through chunks predicted as strikes for quick review.


**Output Format:**
The tool saves a CSV file with the original data plus a `behavior` column containing single-letter codes:
- `s` - Still
- `l` - Locomotion
- `t` - Strike (Note: lowercase 't', not uppercase)
- `u` - Uncertain

Each 5-second chunk gets the same label applied to all its samples (125 rows in the output).

**Tips for Efficient Labeling:**
- Use keyboard shortcuts instead of clicking buttons for faster labeling
- Enable classifier predictions to pre-label the dataset, then quickly review with 'D' key
- Use the zoom buttons if plots are too zoomed in or out
- Use "Show Context" to see surrounding chunks if you're uncertain about a classification
- Adjust chunk size (50-375 samples) using the spinbox if needed for your specific use case
- The save status indicator shows when you have unsaved changes
- The tool prompts to save if you try to close with unsaved work

**What if I make a mistake?**
- Press `A` to go back to the previous chunk
- Reclassify it with the correct number (1-4) - the new label overwrites the old one
- If you've already moved forward many chunks, use the "Go to chunk" feature to jump back
- You can also click "Load Progress" to reload your `_labeled.csv` file and continue from any point

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
  -c 20                         # optional: chunk size in samples (default 20)
  -v                            # optional: verbose logging
```

**Example:**
If you are on the terminal inside of the `snake_behavior` folder, you should be able to run the training via this command:
```bash
python python/train.py -i ./training_data/labeled_data_0p8s.csv -c 20 -v
```
After running, you’ll find:
- `./classifier/training_stats.json` (the extracted information from the training)
- Plots in `./plots/` (one 2D & 3D PDF per behavior)

---

## Evaluator (`evaluate.py`)

**Purpose:**
- Load a **trained** classifier JSON (or auto-train if missing). #TODO: test if auto-training works (it does not as the args aren't passed to the `Trainer` class)
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
  --supervised                      # prompt on borderline chunks during evaluation
  --borderline-threshold 0.1        # relative margin under which a chunk is “borderline”
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

## Validator (validate.py)

Purpose:
- Load a trained classifier JSON
- Load labeled CSV with columns [accX, accY, accZ, behavior]
- Classify via the same Classifier logic (with optional --supervised mode)
- Compute and log overall validation accuracy
- Append validation_accuracy to the JSON model file

Usage:
```bash
python validate.py \
  -m classifier/training_stats.json \   # your model JSON
  -l training_data/labeled_data_5p0s.csv \
  -c 20 \                               # chunk size
  –-supervised \                        # optional live prompting
  –-borderline-threshold 0.1            # optional sensitivity
```

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

---

## Appendix: Beginner's Guide

### Never Used Python or Terminal Before?

Don't worry! This guide will walk you through everything step-by-step.

### What is the Terminal?

The **Terminal** (also called Command Line or Command Prompt) is a text-based way to interact with your computer. Instead of clicking on folders and files, you type commands.

**How to open Terminal:**
- **Mac:** Press `Cmd + Space`, type "Terminal", and press Enter
- **Linux:** Press `Ctrl + Alt + T`
- **Windows:** This guide is for Mac/Linux, but Windows users can use Git Bash or WSL

### Basic Terminal Commands

Here are the essential commands you'll need:

```bash
pwd                    # Shows your current location (Print Working Directory)
ls                     # Lists files and folders in current location
cd folder_name         # Changes directory (moves into a folder)
cd ..                  # Goes up one folder level
cd ~                   # Goes to your home directory
```

**Example:**
```bash
pwd                              # Shows: /Users/yourname
cd Downloads                     # Move into Downloads folder
pwd                              # Shows: /Users/yourname/Downloads
ls                               # Shows all files in Downloads
cd snake_behavior                # Move into the project folder
```

### Your First Time: Complete Walkthrough

**Step 1: Download the project folder**
```bash
cd ~/Desktop/                    # Change this to wherever you want to save the project
git clone https://github.com/tbaer-umich/snake_behavior.git
cd snake_behavior               
pwd                              # Verify you're in the right place
```

**Step 2: Run setup (ONE TIME ONLY)**
```bash
./setup.sh
```
This installs everything you need. You only do this once.

**Step 3: Activate the environment (EVERY TIME)**
```bash
source venv/bin/activate
```
You'll see `(venv)` appear at the start of your terminal line. This means it worked!

**Step 4: Start the labeler**
```bash
python python/labeler.py
```
A window will open!

**Step 5: Use the labeler**
1. Click "Load Data" and select your CSV file
2. Look at the plots showing accelerometer data
3. Press `1` (Still), `2` (Locomotion), `3` (Strike), or `4` (Uncertain)
4. Press `D` to move to the next chunk
5. The tool auto-saves every 25 labels
6. When done, close the window - your work is saved as `<filename>_labeled.csv`

**Common Issues:**
- **"command not found"**: You might be in the wrong folder. Use `pwd` to check, and `cd` to navigate
- **"No such file or directory"**: The path you typed doesn't exist. Use `ls` to see what's available
- **Forgot to activate environment**: You'll see an import error. Run `source venv/bin/activate` first
- **Permission denied on setup.sh**: Run `chmod +x setup.sh` first, then try again

**Need to start over?**
Every time you open a new Terminal window, you need to:
1. Navigate to the project: `cd ~/Desktop/snake_behavior`
2. Activate environment: `source venv/bin/activate`
3. Run the tool: `python python/labeler.py`

That's it! With these basics, you're ready to start labeling snake behavior data.

