# EKPA Programming — Aspect-Based Sentiment Analysis

Final programming assignment for the Programming course at EKPA (University of Athens).

## Overview

This project works with the **SemEval 2016 Task 5** restaurant review dataset (`ABSA16_Restaurants_Train_SB1_v2.xml`) to perform Aspect-Based Sentiment Analysis (ABSA). The dataset contains restaurant reviews annotated with aspect categories, opinion targets, and sentiment polarities at the sentence level.

The main workflow:
1. Split the full XML dataset into 10 smaller parts (each containing 35 reviews).
2. Convert each XML part into a flat CSV for easier analysis.

## Project Structure

```
.
├── data/
│   ├── ABSA16_Restaurants_Train_SB1_v2.xml   # Original full dataset
│   ├── part1.xml – part10.xml                 # Split XML parts
│   └── part1.csv – part10.csv                 # Corresponding CSV files
├── split.py        # Splits the XML dataset into equal-sized chunks
├── utils.py        # Parses XML parts into pandas DataFrames; saves CSVs
├── train.py        # trains models and saves them to disk
├── test.py         # evaluates a saved model on a partial xml file
├── pyproject.toml
└── README.md
```

## Setup

Requires Python ≥ 3.14. Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

### Split the dataset

```bash
uv run split.py
```

This reads `data/ABSA16_Restaurants_Train_SB1_v2.xml` and writes 10 XML part files to `data/`.

### Convert XML parts to CSV

```bash
uv run utils.py
```

This writes 10 csv files to `data/`.

