from pathlib import Path

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
METRICS_DIR = PROJECT_DIR / "metrics"
MODELS_DIR = PROJECT_DIR / "models"

FINAL_RESULTS = PROJECT_DIR / "final_results.json"
DATASET_STATS = PROJECT_DIR / "stats.json"
FULL_DATASET_PATH = DATA_DIR / "ABSA16_Restaurants_Train_SB1_v2.xml"