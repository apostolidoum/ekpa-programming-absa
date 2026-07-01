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