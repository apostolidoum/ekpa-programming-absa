import json
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt

from constants import DATASET_STATS, FINAL_RESULTS, PROJECT_DIR, full_dataset
from utils import concatenate_data


def group_small_slices(counter_obj, threshold_percent=0.03):
    """
    Groups items in a Counter into 'Other' if they fall below the threshold_percent.
    """
    total = sum(counter_obj.values())
    main_categories = {}
    other_count = 0

    # Sort from largest to smallest first
    for key, count in counter_obj.most_common():
        # Check if the slice is larger than the threshold (e.g., 3%)
        if count / total >= threshold_percent:
            main_categories[key] = count
        else:
            other_count += count

    # Only add 'Other' if there were actually small slices grouped
    if other_count > 0:
        main_categories["Other"] = other_count

    return main_categories


def make_dataset_stats_plots(path=DATASET_STATS):
    data = json.load(open(path, "r"))

    polarities = pd.Series(data["Polarity Distribution"])
    ccat = pd.Series(data["Composite Categories"])
    ucat = pd.Series(data["Category Unit Count"])

    for name, characteristic in zip(
        ("Polarity Distribution", "Aspect Categories", "Category Unit Count"),
        (polarities, ccat, ucat),
    ):
        x = 5 if name == "Polarity Distribution" else 10
        plt.figure(1, figsize=(x, 5))
        plt.bar(characteristic.index, characteristic.values, align="center")
        # Rotate 45 degrees and align the text to the right
        plt.xticks(rotation=45, ha="right")
        plt.title(name)
        plt.xlabel("Label")
        plt.ylabel("Count")

        # tight_layout ensures the angled labels don't get cut off at the bottom
        plt.tight_layout()
        plt.savefig(str(PROJECT_DIR / f"{name}_plot.png"))

        plt.show()


def make_results_plots(path, dataset_stats=DATASET_STATS, split_category_label=False):
    full_data = json.load(open(dataset_stats, "r"))
    predictions_data = pd.read_csv(path)

    full_category_counts = (
        Counter(full_data["Category Unit Count"])
        if split_category_label
        else Counter(full_data["Composite Categories"])
    )

    wrong_responses = predictions_data[
        predictions_data["truth"] != predictions_data["preds"]
    ]
    wrong_categories = wrong_responses["category"]

    if split_category_label:
        wrong_category_components = [
            com for cat in wrong_categories for com in cat.split("#")
        ]
        wrong_category_counts = Counter(wrong_category_components)
    else:
        wrong_category_counts = Counter(wrong_categories)

    grouped_full = group_small_slices(full_category_counts, threshold_percent=0.024)
    grouped_wrong = group_small_slices(wrong_category_counts, threshold_percent=0.024)

    full_labels = list(grouped_full.keys())
    full_values = list(grouped_full.values())

    wrong_labels = list(grouped_wrong.keys())
    wrong_values = list(grouped_wrong.values())

    full_explode = [0.1 if label == "Other" else 0 for label in full_labels]
    wrong_explode = [0.1 if label == "Other" else 0 for label in wrong_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

    ax1.pie(
        full_values,
        labels=full_labels,
        explode=full_explode,
        autopct="%1.1f%%",
        startangle=90,
        shadow=False,
        rotatelabels=True,
        labeldistance=1,
    )
    # ax1.set_title("Full Category Counts")

    ax2.pie(
        wrong_values,
        labels=wrong_labels,
        explode=wrong_explode,
        autopct="%1.1f%%",
        startangle=90,
        shadow=False,
        rotatelabels=True,
        labeldistance=1,
    )
    # ax2.set_title("Wrong Category Counts")

    plt.suptitle("Category Distribution Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_parts_sizes():
    parts = {}
    for name in full_dataset:
        data = concatenate_data([name])
        parts.update({name: data.shape[0]})
    parts = pd.Series(parts)

    plt.figure(1, figsize=(10, 5))
    plt.bar(parts.index, parts.values, align="center")
    # Rotate 45 degrees and align the text to the right
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Part")
    plt.ylabel("Number of Sentences")
    plt.title("Part Size Distribution")

    # tight_layout ensures the angled labels don't get cut off at the bottom
    plt.tight_layout()

    plt.savefig(str(PROJECT_DIR / f"parts_plot.png"))

    plt.show()


def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i - 0.25, y[i], f"{y[i]:.2f}")


def plot_cross_validation_results_all_models(results_json=FINAL_RESULTS):

    stats = json.load(
        open(
            results_json,
            "r",
        )
    )

    plt.figure(1, figsize=(13, 5))
    names = list(stats.keys())
    accuracies = [metrics[1] for key, metrics in stats.items()]
    plt.bar(names, accuracies, align="center")
    add_labels(names, accuracies)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.tight_layout()

    plt.show()


# plot_cross_validation_results_all_models(r"C:\Users\dioni\ekpa-programming-absa\final_results.json")
make_results_plots(
    r"C:\Users\dioni\ekpa-programming-absa\cross-validation_results\svm_one_hotCWlgram(1, 1)_reduce_f_False_n_components_1000_predictions.csv"
)
# make_dataset_stats_plots()
