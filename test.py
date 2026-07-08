import os
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from constants import METRICS_DIR, full_dataset
from train import (
    svm_one_hot,
)
from utils import (
    concatenate_data,
    get_feature_dimensionality,
    load_model,
    split_features_from_target,
)


def build_model(function, train_set, **kwargs):
    path = function(train_set, **kwargs)
    return path


def plot_conf_matrix(cm, model_type, metrics_dir=METRICS_DIR):
    # 1. Plot the Distribution of Predictions
    plt.figure(figsize=(6, 5))

    # 2. Plot Confusion Matrix (Comparing preds to true labels y)
    labels = [
        "negative",
        "neutral",
        "positive",
    ]  # [l if l else "Broken" for l in model.classes_]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,  # Adjust to your classes
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    os.makedirs(metrics_dir, exist_ok=True)
    plt.savefig(Path(metrics_dir, f"{model_type}_confusion_matrix.png"))

    return cm


def evaluate_model(
    clf: str, test_set: str, dir=METRICS_DIR, lngrams=False, custom_context_window=False
):
    # TODO maybe split ploting logic from accuracy results?

    model_id = Path(dir, Path(clf).stem)
    split_id = str(model_id.stem) + test_set.rstrip(".xml")

    if "models" in PurePath(clf).parts:
        clf: sklearn.pipeline.Pipeline = load_model(clf)
    else:
        clf: sklearn.pipeline.Pipeline = load_model(os.path.join("models", clf))

    df = concatenate_data([test_set])
    key = "one-hot" if "preprocessor" in clf.named_steps else "text_f"

    print(f"Model Type: {split_id}")

    X, y = split_features_from_target(
        df, key, lngrams=lngrams, target_context_window=custom_context_window
    )

    preds = clf.predict(X)
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=1)

    cm = confusion_matrix(y, preds)
    plot_conf_matrix(cm, split_id, model_id)

    clr = classification_report(y, preds, output_dict=True)

    return clr["accuracy"], cm, (y, preds), clr


def evaluate_embeds_model(clf: str, test_set: str, dir=METRICS_DIR):
    model_id = Path(dir, Path(clf).stem)
    split_id = str(model_id.stem) + test_set.rstrip(".xml")

    if "models" in PurePath(clf).parts:
        clf = load_model(clf)
    else:
        clf = load_model(os.path.join("models", clf))

    df = concatenate_data([test_set])
    df = df.dropna()

    formatted_texts = ("Aspect: " + df["category"] + " Review: " + df["text"]).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(formatted_texts)
    y = df["polarity"]

    preds = clf.predict(X)
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=1)

    cm = confusion_matrix(y, preds)
    plot_conf_matrix(cm, split_id, model_id)

    clr = classification_report(y, preds, output_dict=True)

    return clr["accuracy"], cm, (y, preds), clr


def main():
    print("Example of using the test code.")
    test_files = ["part2.xml"]
    train_files = [f for f in full_dataset if f not in test_files]

    print(f"Training a model on {train_files}")
    model_path = svm_one_hot(files_to_use=train_files, reduce_f=True)

    acc, _, _, res = evaluate_model(model_path, "part2.xml")
    print(f"Results on {test_files} is \n{pd.DataFrame(res)}")
    print(get_feature_dimensionality(model_path))


if __name__ == "__main__":
    main()

    # Testing the evaluate embeds model
    # test_files = ["part2.xml"]
    # train_files = [f for f in full_dataset if f not in test_files]

    # print(f"Training a model on {train_files}")
    # model_path = embeds(files_to_use=train_files)

    # acc, _, _, res = evaluate_embeds_model(model_path, "part2.xml")
    # print(f"Results on {test_files} is \n{pd.DataFrame(res)}")
