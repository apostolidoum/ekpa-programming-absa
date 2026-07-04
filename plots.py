import json
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from utils import concatenate_data
from constants import PROJECT_DIR, full_dataset


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


def make_dataset_stats_plots(path=PROJECT_DIR / "stats.json"):
    data = json.load(open(path, "r"))

    polarities = pd.Series(data["Polarity Distribution"])
    ccat = pd.Series(data["Composite Categories"])
    ucat = pd.Series(data["Category Unit Count"])

    for name, characteristic in zip(
        ("Pol_", "Comp_", "Unit_"), (polarities, ccat, ucat)
    ):
        x = 5 if name == "Pol_" else 10
        plt.figure(1, figsize=(x, 5))
        plt.bar(characteristic.index, characteristic.values, align="center")
        # Rotate 45 degrees and align the text to the right
        plt.xticks(rotation=45, ha="right")

        # tight_layout ensures the angled labels don't get cut off at the bottom
        plt.tight_layout()
        plt.show()
        plt.savefig(str(PROJECT_DIR / f"{name}plot.png"))


def make_results_plots(
    path=PROJECT_DIR / "predictions.csv", dataset_stats=PROJECT_DIR / "stats.json"
):
    full_data = json.load(open(dataset_stats, "r"))
    predictions_data = pd.read_csv(path)

    # full_category_counts = Counter(full_data["Composite Categories"])
    full_category_counts = Counter(full_data["Category Unit Count"])

    wrong_responses = predictions_data[
        predictions_data["truth"] != predictions_data["preds"]
    ]
    wrong_categories = wrong_responses["category"]
    wrong_category_components = [
        com for cat in wrong_categories for com in cat.split("#")
    ]
    wrong_category_counts = Counter(wrong_category_components)

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

    # --- Global Formatting ---
    plt.suptitle("Full - Category Distribution Comparison - Wrong", fontsize=16)
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

    # tight_layout ensures the angled labels don't get cut off at the bottom
    plt.tight_layout()
    plt.show()
    plt.savefig(str(PROJECT_DIR / f"parts_plot.png"))


def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], f"{y[i]:.2f}")


def plot_cross_validation_results_all_models():
    stats = json.load(
        open(
            "/Users/maria/Documents/DPMS/2nd_semester/programming/ekpa-programming-absa/final_results.json",
            "r",
        )
    )
    names = list(stats.keys())
    accuracies = [metrics[1] for key, metrics in stats.items()]
    plt.bar(names, accuracies, align="center")
    add_labels(names, accuracies)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


plot_cross_validation_results_all_models()
