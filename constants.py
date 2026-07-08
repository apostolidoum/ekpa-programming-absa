from pathlib import Path

import spacy
from spacy.cli import download
from spacy.util import is_package

full_dataset = [
    "part1.xml",
    "part2.xml",
    "part3.xml",
    "part4.xml",
    "part5.xml",
    "part6.xml",
    "part7.xml",
    "part8.xml",
    "part9.xml",
    "part10.xml",
]

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
CROSS_VAL_PREDS = PROJECT_DIR / "cross-validation_results"
METRICS_DIR = PROJECT_DIR / "metrics"
MODELS_DIR = PROJECT_DIR / "models"
HISTORY_DIR = PROJECT_DIR / "history"
FEATURES_DIR = PROJECT_DIR / "features"

FINAL_RESULTS = METRICS_DIR / "final_results.json"
DATASET_STATS = DATA_DIR / "stats.json"
FULL_DATASET_PATH = DATA_DIR / "ABSA16_Restaurants_Train_SB1_v2.xml"

model_name = "en_core_web_sm"

if not is_package(model_name):
    print(f"Model '{model_name}' not found. Downloading...")
    download(model_name)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CROSS_VAL_PREDS.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
