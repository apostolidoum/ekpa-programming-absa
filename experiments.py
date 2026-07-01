from test import evaluate_model, plot_conf_matrix
from constants import full_dataset
from train import (
    svm_one_hot,
    svm_text_features,
    logistic_regression_one_hot,
    logistic_regression_text_features
    )
from utils import get_model_name_from_path


def cross_validate(model_builder,
                    reduce_f=False,
                    n_components=1000,
                    **kwargs):

    results = []
    confs = []
    m_path = ""

    for test_set in full_dataset:
        train_set = [i for i in full_dataset if i!=test_set]

        model_path = model_builder(train_set, reduce_f=reduce_f, n_components=n_components, **kwargs)
        m_path = model_path
        acc, cm = evaluate_model(model_path, test_set)
        results.append(acc)
        confs.append(cm)

    avg = sum(results)/len(results)
    total_confusion = sum(confs)
    plot_conf_matrix(total_confusion, m_path.stem)

    return results, avg

print(cross_validate(svm_text_features))