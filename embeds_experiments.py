from train import embeds
from constants import full_dataset, CROSS_VAL_PREDS, METRICS_DIR
from test import evaluate_embeds_model, plot_conf_matrix
from utils import concatenate_data
import pandas as pd
from experiments import save_results


def cross_validate(model_builder=embeds):
    results = []
    accs = []
    confs = []
    m_path = ""
    predictions = []

    for test_set in full_dataset:
        train_set = [i for i in full_dataset if i != test_set]

        model_path = model_builder(train_set)
        m_path = model_path
        acc, cm, (y, preds), report = evaluate_embeds_model(model_path, test_set)
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
    predictions.to_csv(
        CROSS_VAL_PREDS / "embeds_model_predictions.csv",
        index=False,
    )

    return results, avg.to_dict()


def main():
    final_results = {}

    models = [{"EMBEDS": {"model_builder": embeds}}]

    for model_entry in models:
        print(model_entry.items())
        model = list(model_entry.values())[0]
        key = list(model_entry.keys())[0]

        final_results.update({key: cross_validate(**model)})
        final_results = save_results(final_results, METRICS_DIR / "embeds_results.json")


if __name__ == "__main__":
    main()
