import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from constants import full_dataset
from train import (
    svm_one_hot,
)
from utils import (
    concatenate_data,
    get_feature_dimensionality,
    has_preprocessor,
    load_model,
    split_features_from_target,
)


def build_model(function, train_set, **kwargs):

    path = function(train_set, **kwargs)

    return path


def plot_conf_matrix(y, yhat, model, model_type, metrics_dir="metrics"):
    # 1. Plot the Distribution of Predictions
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.countplot(x=yhat, palette="viridis")
    plt.title("Predicted Class Distribution")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Count")

    # 2. Plot Confusion Matrix (Comparing preds to true labels y)
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y, yhat)
    labels = [l if l else "Broken" for l in model.classes_]
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
    plt.savefig(os.path.join(metrics_dir, f"{model_type}_confusion_matrix.png"))

    return plt


def evaluate_model(clf, *test_set: str):
    # TODO maybe split ploting logic from accuracy results?

    if isinstance(clf, str):
        if "models/" in clf:
            clf = load_model(clf)
        else:
            clf = load_model(os.path.join("models", clf))

    df = concatenate_data(list(test_set))
    key = "one-hot" if has_preprocessor(clf) else "text_f"

    classifier_step = clf.named_steps["classifier"]
    model_type_str = type(classifier_step).__name__ + "_" + key
    print(f"Model Type: {model_type_str}")

    X, y = split_features_from_target(df, key)

    preds = clf.predict(X)
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=1)

    cm = plot_conf_matrix(y, preds, clf, model_type_str)
    # cm.show()

    clr = classification_report(y, preds, output_dict=True)

    return clr["accuracy"]


def main():
    print("Example of using the test code.")
    test_files = ["part1.xml"]
    train_files = [f for f in full_dataset if f not in test_files]

    print(f"Training a model on {train_files}")
    svm_one_hot(files_to_use=train_files)
    model = load_model(
        "models/svm_onehot_ngram_(1, 3)_max_iter_1000_C_1.0_reduce_f_False_n_components_1000.pkl"
    )

    acc = evaluate_model(model, "part1.xml")
    print(f"Accuracy on {test_files} is {acc}")
    print(get_feature_dimensionality(model))


if __name__ == "__main__":
    main()
