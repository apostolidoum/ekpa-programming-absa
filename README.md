# EKPA Programming — Aspect-Based Sentiment Analysis

Final programming assignment for the Programming course at EKPA (University of Athens).

## Overview

This project works with the **SemEval 2016 Task 5** restaurant review dataset (`ABSA16_Restaurants_Train_SB1_v2.xml`) to perform Aspect-Based Sentiment Analysis (ABSA). The dataset contains restaurant reviews annotated with aspect categories, opinion targets, and sentiment polarities at the sentence level.

The main workflow:
1. Split the full XML dataset into 10 smaller parts (each containing 35 reviews).
2. Convert each XML part into a flat CSV for easier analysis.

## Project Structure

```
PROJECT_DIR/
├── data/                                 # Raw and processed datasets
│   ├── ABSA16_Restaurants_Train_SB1_v2.xml
│   ├── part1-10.xml
│   ├── part1-10.csv
│   └── stats.json                        # dataset stats
├── cross-validation_results/             # CV predictions - truth maps 
├── metrics/                              # Metrics output folder
│   ├── ngram_...n_components_1000_/      # Folders containing per-fold confusion matrices
│   ├── ...confusion_matrix.png           # Averaged confusion matrices across folds per configuration
│   ├── final_results.json                # Experiment result metrics
│   ├── embeds_results.json
│   ├── enhanced_exp_final_results.json
│   └── ablation_final_results.json                        
├── models/                               # Saved model pickles
├── history/                              # Empty, mostly for easy manual use
├── features/                             # Extracted features per model / class, sorted by importance
├── split.py                              # Splits the XML dataset into equal-sized chunks
├── utils.py                              # Parses XML parts into pandas DataFrames; saves CSVs  
├── train.py                              # Trains models and saves them to disk  
├── test.py                               # Evaluates a saved model on a partial xml file
├── constants.py                          # Centralized path / globals management
├── experiments.py                        # Start stage 1 experiment.
├── embeds_experiments.py                 # Complete stage 1 experiment.
├── ablation.py                           # Run stage 2 experiment.
├── enhanced_experiments.py               # Run stage 3 experiment.
├── feature_importance.py                 # Feature importance stats
├── plots.py                              # Contains functions used to make plots
└── pyproject.toml
```

## Setup

Requires Python ≥ 3.14. Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage
****

### Prepare project environment

```bash
uv run constants.py
```
This creates the necessary directory structure and globals required for the code to run smoothly. \
It also downloads the necessary `en_core_web_sm` package from spaCy, which is required for lemmatization.

****
### Split the dataset

```bash
uv run split.py
```
This reads `data/ABSA16_Restaurants_Train_SB1_v2.xml` and writes 10 XML part files.
****
### Convert XML parts to CSV and save dataset statistics
**Optional step**: Execute to write a json file containing statistical information on the full dataset, and save xml parts to csv format.

```bash
uv run utils.py
```

This writes 10 csv files as well as `stats.json` to `data/`.
****
### First stage of experiments

Evaluate different n-gram range (1, 3) model configurations using a 10-fold cross validation setup. 

**Parameters examined**:
1. Classifier (Log. Reg. or Linear SVM)
2. Aspect Encoding (as a text feature [TF] or OH vector)
3. Feature Extraction (k=200 and k=1000)
4. Embedding-SVM model baseline

```bash
uv run experiments.py
uv run embeds_experiments.py
```

These output confusion matrices and full metrics reports (`final_results.json, embeds_final_results.json`) to `metrics/`.
Files mapping predictions to true labels for all folds are saved to `cross-validation-results/`.
****
### Second stage of experiments

Compare training on n-gram ranges (1, 1) and (2, 3), using a 10-fold cross validation setup. 

```bash
uv run ablation.py
```

This outputs confusion matrices and full metrics report (`ablation_final_results.json`) to `metrics/`.
Files mapping predictions to true labels for all folds are saved to `cross-validation-results/`.

****
### Third stage of experiments

Examine the impact of two preprocessing methods on the final model, using a 10-fold cross validation setup. 

**Parameters examined**:
1. Target Context Window Application (CCW of size 6 around target token)
2. Lemmatization (lgrams instead of ngrams)

```bash
uv run enhanced_experiments.py
```

This outputs confusion matrices and full metrics report (`enhanced_exp_final_results.json`) to `metrics/`.
Files mapping predictions to true labels for all folds are saved to `cross-validation-results/`.

****
### Feature Inspection

Examine the most important features used by a model. 

**Arguments**:
1. **--class-label**: Class you wish to get results for. \
_Options: 0, 1, 2 = negative, neutral, positive (respectively)_

2. **--model-tag** (optional): Simple string tag to disambiguate features file.
3. **--model-path** (optional, with default): Path to target .pkl file. \
_Default: Path to the BASE SVMOH (1, 3) model._
4. **-v** (optional): Verbose mode, prints top 50 features to terminal

```bash
uv run feature_importance.py --class-label <...> --model-tag <OPT> --model-path <OPT> [-v]
```

This outputs a `{model-tag}_Class_{int_class-label}_feature_importance.csv` to `features/`.
