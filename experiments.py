from pathlib import Path

import pandas as pd

from test import evaluate_model, plot_conf_matrix
from constants import full_dataset, PROJECT_DIR, CROSS_VAL_PREDS, FINAL_RESULTS
from train import (
    svm_one_hot,
    svm_text_features,
    logistic_regression_one_hot,
    logistic_regression_text_features,
)
import json

from utils import concatenate_data, xml_to_dataframe


def cross_validate(
    model_builder=svm_one_hot,
    reduce_f=False,
    n_components=1000,
    ngram_range=(1, 3),
    **kwargs,
):

    results = []
    accs = []
    confs = []
    m_path = ""
    predictions = []

    for test_set in full_dataset:
        train_set = [i for i in full_dataset if i != test_set]

        model_path = model_builder(
            train_set,
            reduce_f=reduce_f,
            n_components=n_components,
            ngram_range=ngram_range,
            **kwargs,
        )

        m_path = model_path
        acc, cm, (y, preds), report = evaluate_model(model_path, test_set, **kwargs)
        print(report)

        categories = concatenate_data([test_set])["category"].dropna().values

        print(
            [
                len(i)
                for i in ([test_set in range(len(categories))], categories, y, preds)
            ]
        )

        predictions.append(
            pd.DataFrame(
                {
                    "test": [test_set for _ in range(len(categories))],
                    "category": categories,
                    "truth": y,
                    "preds": preds,
                }
            )
        )

        print(type(report), report)
        results.append(report)
        accs.append(acc)
        confs.append(cm)

    efficient_results = [pd.DataFrame(i) for i in results]
    avg = sum(efficient_results) / len(efficient_results)
    total_confusion = sum(confs)
    plot_conf_matrix(total_confusion, m_path.stem)
    predictions = pd.concat(predictions)

    lgram_tag = "lgram" if kwargs.get('lngrams') == True else ""
    window_tag = "CW" if kwargs.get('custom_context_window') == True else ""

    predictions.to_csv(
        CROSS_VAL_PREDS
        / f"{model_builder.__name__}{window_tag}{lgram_tag}{ngram_range}_reduce_f_{reduce_f}_n_components_{n_components}_predictions.csv",
        index=False,
    )

    return results, avg.to_dict()


def save_results(results, target=FINAL_RESULTS):
    if isinstance(results, pd.DataFrame):
        results = results.to_json(orient="index")
    json.dump(results, open(target, "w"))
    results = json.load(open(target))

    return results


def main():

    if FINAL_RESULTS.exists():
        try:
            final_results = json.load(open(FINAL_RESULTS))
        except json.decoder.JSONDecodeError as e:
            print(
                f"JSON incorrectly formatted or empty. Starting new results log. \n{e}"
            )
            final_results = {}
    else:
        final_results = {}

    models = [
        {
            "BASE_LRTF_N1,3": {
                "model_builder": logistic_regression_text_features,
                "reduce_f": False,
                "ngram_range": (1, 3),
            }
        },
        {
            "BASE_LROH_N1,3": {
                "model_builder": logistic_regression_one_hot,
                "reduce_f": False,
                "ngram_range": (1, 3),
            }
        },
        {
            "BASE_SVMTF_N1,3": {
                "model_builder": svm_text_features,
                "reduce_f": False,
                "ngram_range": (1, 3),
            }
        },
        {
            "BASE_SVMOH_N1,3": {
                "model_builder": svm_one_hot,
                "reduce_f": False,
                "ngram_range": (1, 3),
            }
        },
        {
            "k1000_LRTF_N1,3": {
                "model_builder": logistic_regression_text_features,
                "reduce_f": True,
                "n_components": 1000,
                "ngram_range": (1, 3),
            }
        },
        {
            "k1000_LROH_N1,3": {
                "model_builder": logistic_regression_one_hot,
                "reduce_f": True,
                "n_components": 1000,
                "ngram_range": (1, 3),
            }
        },
        {
            "k1000_SVMTF_N1,3": {
                "model_builder": svm_text_features,
                "reduce_f": True,
                "n_components": 1000,
                "ngram_range": (1, 3),
            }
        },
        {
            "k1000_SVMOH_N1,3": {
                "model_builder": svm_one_hot,
                "reduce_f": True,
                "n_components": 1000,
                "ngram_range": (1, 3),
            }
        },
        {
            "k200_LRTF_N1,3": {
                "model_builder": logistic_regression_text_features,
                "reduce_f": True,
                "n_components": 200,
                "ngram_range": (1, 3),
            }
        },
        {
            "k200_LROH_N1,3": {
                "model_builder": logistic_regression_one_hot,
                "reduce_f": True,
                "n_components": 200,
                "ngram_range": (1, 3),
            }
        },
        {
            "k200_SVMTF_N1,3": {
                "model_builder": svm_text_features,
                "reduce_f": True,
                "n_components": 200,
                "ngram_range": (1, 3),
            }
        },
        {
            "k200_SVMOH_N1,3": {
                "model_builder": svm_one_hot,
                "reduce_f": True,
                "n_components": 200,
                "ngram_range": (1, 3),
            }
        },
    ]

    for model_entry in models:
        print(model_entry.items())
        model = list(model_entry.values())[0]
        key = list(model_entry.keys())[0]

        final_results.update({key: cross_validate(**model)})
        print(type(final_results), final_results)
        final_results = save_results(final_results)


if __name__ == "__main__":
    main()

# print(cross_validate(svm_text_features))
