from test import evaluate_model, plot_conf_matrix
from constants import full_dataset
from train import (
    svm_one_hot,
    svm_text_features,
    logistic_regression_one_hot,
    logistic_regression_text_features
    )
import json

def cross_validate(model_builder,
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


def main():

    final_results = {}

    final_results.update({"BASE_LRTF_N1,3": cross_validate(logistic_regression_text_features, reduce_f=False, ngram_range=(1, 3))})       # BASE_LRTF_N1,3
    final_results.update({"BASE_LROH_N1,3": cross_validate(logistic_regression_one_hot, reduce_f=False, ngram_range=(1, 3))})             # BASE_LROH_N1,3
    final_results.update({"BASE_SVMTF_N1,3": cross_validate(svm_text_features, reduce_f=False, ngram_range=(1, 3))})                       # BASE_SVMTF_N1,3
    final_results.update({"BASE_SVMOH_N1,3": cross_validate(svm_one_hot, reduce_f=False, ngram_range=(1, 3))})                             # BASE_SVMOH_N1,3

    final_results.update({"k1000_LRTF_N1,3": cross_validate(logistic_regression_text_features, reduce_f=True, n_components=1000, ngram_range=(1, 3))}) # k 1000 LRTF
    final_results.update({"k1000_LROH_N1,3": cross_validate(logistic_regression_one_hot, reduce_f=True, n_components=1000      , ngram_range=(1, 3))}) # k 1000 LROH
    final_results.update({"k1000_SVMTF_N1,3":cross_validate(svm_text_features, reduce_f=True, n_components=1000                , ngram_range=(1, 3))}) # k 1000 SVMTF
    final_results.update({"k1000_SVMOH_N1,3":cross_validate(svm_one_hot, reduce_f=True, n_components=1000                      , ngram_range=(1, 3))}) # k 1000 SVMOH

    final_results.update({"k2000_LRTF_N1,3": cross_validate(logistic_regression_text_features, reduce_f=True, n_components=2000, ngram_range=(1, 3))})  # k 2000 LRTF
    final_results.update({"k2000_LROH_N1,3": cross_validate(logistic_regression_one_hot, reduce_f=True, n_components=2000      , ngram_range=(1, 3))})  # k 2000 LROH
    final_results.update({"k2000_SVMTF_N1,3":cross_validate(svm_text_features, reduce_f=True, n_components=2000                , ngram_range=(1, 3))})  # k 2000 SVMTF
    final_results.update({"k2000_SVMOH_N1,3":cross_validate(svm_one_hot, reduce_f=True, n_components=2000                      , ngram_range=(1, 3))})  # k 2000 SVMOH

    final_results.update({"BASE_LRTF_N1,1": cross_validate(logistic_regression_text_features, reduce_f=False, ngram_range=(1, 1))}) # Base LRTF
    final_results.update({"BASE_LROH_N1,1": cross_validate(logistic_regression_one_hot, reduce_f=False      , ngram_range=(1, 1))}) # BASE LROH
    final_results.update({"BASE_SVMTF_N1,1":cross_validate(svm_text_features, reduce_f=False                , ngram_range=(1, 1))}) # BASE SVMTF
    final_results.update({"BASE_SVMOH_N1,1":cross_validate(svm_one_hot, reduce_f=False                      , ngram_range=(1, 1))}) # BASE SVMOH

    final_results.update({"k1000_LRTF_N1,1": cross_validate(logistic_regression_text_features, reduce_f=True, n_components=1000, ngram_range=(1, 1))}) # k 1000 LRTF
    final_results.update({"k1000_LROH_N1,1": cross_validate(logistic_regression_one_hot, reduce_f=True, n_components=1000      , ngram_range=(1, 1))}) # k 1000 LROH
    final_results.update({"k1000_SVMTF_N1,1":cross_validate(svm_text_features, reduce_f=True, n_components=1000                , ngram_range=(1, 1))}) # k 1000 SVMTF
    final_results.update({"k1000_SVMOH_N1,1":cross_validate(svm_one_hot, reduce_f=True, n_components=1000                      , ngram_range=(1, 1))}) # k 1000 SVMOH

    final_results.update({"k2000_LRTF_N1,1": cross_validate(logistic_regression_text_features, reduce_f=True, n_components=2000, ngram_range=(1, 1))})  # k 2000 LRTF
    final_results.update({"k2000_LROH_N1,1": cross_validate(logistic_regression_one_hot, reduce_f=True, n_components=2000      , ngram_range=(1, 1))})  # k 2000 LROH
    final_results.update({"k2000_SVMTF_N1,1":cross_validate(svm_text_features, reduce_f=True, n_components=2000                , ngram_range=(1, 1))})  # k 2000 SVMTF
    final_results.update({"k2000_SVMOH_N1,1":cross_validate(svm_one_hot, reduce_f=True, n_components=2000                      , ngram_range=(1, 1))})  # k 2000 SVMOH

    json.dump(final_results, open("final_results.json", "w"))

if __name__ == "__main__":
    main()

#print(cross_validate(svm_text_features))