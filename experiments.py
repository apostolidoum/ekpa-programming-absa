from pathlib import Path

from test import evaluate_model, plot_conf_matrix
from constants import full_dataset, PROJECT_DIR
from train import (
    svm_one_hot,
    svm_text_features,
    logistic_regression_one_hot,
    logistic_regression_text_features
    )
import json

def cross_validate( model_builder=svm_one_hot,
                    reduce_f=False,
                    n_components=1000,
                    ngram_range=(1, 3),
                    **kwargs):

    results = []
    confs = []
    m_path = ""

    for test_set in full_dataset:
        train_set = [i for i in full_dataset if i != test_set]

        model_path = model_builder(train_set, reduce_f=reduce_f, n_components=n_components, ngram_range=ngram_range, **kwargs)
        m_path = model_path
        acc, cm = evaluate_model(model_path, test_set)
        results.append(acc)
        confs.append(cm)

    avg = sum(results)/len(results)
    total_confusion = sum(confs)
    plot_conf_matrix(total_confusion, m_path.stem)

    return results, avg


def save_results(results):

    json.dump(results, open("final_results.json", "w"))
    results = json.load(open("final_results.json"))

    return results


def main():

    if Path(PROJECT_DIR, "final_results.json").exists():
        final_results = json.load(open("final_results.json"))

    else:
        final_results = {}

    models = [
        {"BASE_LRTF_N1,3": {"model_builder": logistic_regression_text_features, "reduce_f": False,  "ngram_range": (1, 3)}},
        {"BASE_LROH_N1,3": {"model_builder": logistic_regression_one_hot, "reduce_f": False, "ngram_range": (1, 3)}},
        {"BASE_SVMTF_N1,3": {"model_builder": svm_text_features, "reduce_f": False, "ngram_range": (1, 3)}},
        {"BASE_SVMOH_N1,3": {"model_builder": svm_one_hot, "reduce_f": False, "ngram_range": (1, 3)}},
        {"k1000_LRTF_N1,3": {"model_builder": logistic_regression_text_features, "reduce_f": True, "n_components": 1000, "ngram_range": (1, 3)}},
        {"k1000_LROH_N1,3": {"model_builder": logistic_regression_one_hot, "reduce_f": True, "n_components": 1000, "ngram_range": (1, 3)}},
        {"k1000_SVMTF_N1,3": {"model_builder": svm_text_features, "reduce_f": True, "n_components": 1000, "ngram_range": (1, 3)}},
        {"k1000_SVMOH_N1,3": {"model_builder": svm_one_hot, "reduce_f": True, "n_components": 1000, "ngram_range": (1, 3)}},
        {"k200_LRTF_N1,3": {"model_builder": logistic_regression_text_features, "reduce_f": True, "n_components": 200, "ngram_range": (1, 3)}},
        {"k200_LROH_N1,3": {"model_builder": logistic_regression_one_hot, "reduce_f": True, "n_components": 200, "ngram_range": (1, 3)}},
        {"k200_SVMTF_N1,3": {"model_builder": svm_text_features, "reduce_f": True, "n_components": 200, "ngram_range": (1, 3)}},
        {"k200_SVMOH_N1,3": {"model_builder": svm_one_hot, "reduce_f": True, "n_components": 200, "ngram_range": (1, 3)}},
        {"BASE_LRTF_N1,1": {"model_builder": logistic_regression_text_features, "reduce_f": False, "ngram_range": (1, 1)}},
        {"BASE_LROH_N1,1": {"model_builder": logistic_regression_one_hot, "reduce_f": False, "ngram_range": (1, 1)}},
        {"BASE_SVMTF_N1,1": {"model_builder": svm_text_features, "reduce_f": False, "ngram_range": (1, 1)}},
        {"BASE_SVMOH_N1,1": {"model_builder": svm_one_hot, "reduce_f": False, "ngram_range": (1, 1)}},
        {"k1000_LRTF_N1,1": {"model_builder": logistic_regression_text_features, "reduce_f": True, "n_components": 1000, "ngram_range": (1, 1)}},
        {"k1000_LROH_N1,1": {"model_builder": logistic_regression_one_hot, "reduce_f": True, "n_components": 1000, "ngram_range": (1, 1)}},
        {"k1000_SVMTF_N1,1": {"model_builder": svm_text_features, "reduce_f": True, "n_components": 1000, "ngram_range": (1, 1)}},
        {"k1000_SVMOH_N1,1": {"model_builder": svm_one_hot, "reduce_f": True, "n_components": 1000, "ngram_range": (1, 1)}},
        {"k200_LRTF_N1,1": {"model_builder": logistic_regression_text_features, "reduce_f": True, "n_components": 200, "ngram_range": (1, 1)}},
        {"k200_LROH_N1,1": {"model_builder": logistic_regression_one_hot, "reduce_f": True, "n_components": 200, "ngram_range": (1, 1)}},
        {"k200_SVMTF_N1,1": {"model_builder": svm_text_features, "reduce_f": True, "n_components": 200, "ngram_range": (1, 1)}},
        {"k200_SVMOH_N1,1": {"model_builder": svm_one_hot, "reduce_f": True, "n_components": 200, "ngram_range": (1, 1)}}
    ]

    for model_entry in models:
        print(model_entry.items())
        model = list(model_entry.values())[0]
        key = list(model_entry.keys())[0]

        final_results.update({key: cross_validate(**model)})
        final_results = save_results(final_results)


if __name__ == "__main__":
    main()

#print(cross_validate(svm_text_features))