from constants import METRICS_DIR
from experiments import cross_validate, save_results
from train import (
    svm_one_hot,
    svm_text_features,
    )

def main():

    final_results = {}

    models = [
        {"k1000_SVMOH_N1,1": {"model_builder": svm_one_hot, "reduce_f": True, "n_components": 1000,
                              "ngram_range": (1, 1)}},
        {"BASE_SVMOH_N1,1": {"model_builder": svm_one_hot, "reduce_f": False, "n_components": 1000,
                             "ngram_range": (1, 1)}},
        {"k1000_SVMOH_N2,3": {"model_builder": svm_one_hot, "reduce_f": True, "n_components": 1000,
                              "ngram_range": (2, 3)}},
        {"BASE_SVMOH_N2,3": {"model_builder": svm_one_hot, "reduce_f": False, "n_components": 1000,
                              "ngram_range": (2, 3)}},
    ]

    for model_entry in models:
        print(model_entry.items())
        model = list(model_entry.values())[0]
        key = list(model_entry.keys())[0]

        final_results.update({key: cross_validate(**model)})
        final_results = save_results(final_results, METRICS_DIR / "ablation_final_results.json")

if __name__ == "__main__":
    main()