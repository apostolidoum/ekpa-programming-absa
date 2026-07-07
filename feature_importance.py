import argparse
import numpy as np
import sklearn
from constants import MODELS_DIR, FEATURES_DIR
from utils import *


def get_top_features_before_reduction(
    model: str | Path | sklearn.pipeline.Pipeline, target_class=0
):
    """Target classes: 0=Negative, 1=Neutral, 2=Positive"""

    if isinstance(model, str | Path):
        model = load_model(model)

    classifier = model.named_steps["classifier"]

    if "reducer" in model.named_steps:
        print("Your model has a reducer attached. Forwarding to appropriate function.")
        return get_top_features_post_reduction(model, target_class)

    if "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
    else:
        preprocessor = model.named_steps["tfidf"]

    ngrams = preprocessor.get_feature_names_out()
    weights = classifier.coef_[target_class]

    feature_importance = pd.DataFrame({"ngrams": ngrams, "importance": weights})

    top_features = feature_importance.sort_values(
        "importance", key=abs, ascending=False
    )

    return top_features


def get_top_features_post_reduction(
    model: str | Path | sklearn.pipeline.Pipeline, target_class=0
):
    """Target classes: 0=Negative, 1=Neutral, 2=Positive"""

    if isinstance(model, str | Path):
        model = load_model(model)

    try:
        red = model.named_steps["reducer"]
    except KeyError as e:
        print(
            f"Model pipeline does not feature a reducer named step! Forwarding to appropriate function."
        )
        return get_top_features_before_reduction(model, target_class)

    classifier = model.named_steps["classifier"]

    if "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
    else:
        preprocessor = model.named_steps["tfidf"]

    orig_weights = np.dot(classifier.coef_, red.components_)
    all_features = preprocessor.get_feature_names_out()

    df_importance = pd.DataFrame(
        {"feature": all_features, "importance": orig_weights[target_class]}
    )

    top_features = df_importance.sort_values(by="importance", key=abs, ascending=False)

    return top_features


def main():
    parser = argparse.ArgumentParser(description="View feature importance for model.")

    parser.add_argument(
        "--class-label",
        choices=["negative", "positive", "neutral", "0", "1", "2"],
        default="0",
        help="Class label to analyze (default: negative)",
    )

    parser.add_argument(
        "--model-tag",
        type=str,
        default="DEF",
        help="Model id to disambiguate output file name.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=MODELS_DIR
        / "svm_onehot_ngram_(1, 3)_max_iter_1000_C_1-0_reduce_f_False_n_components_1000_.pkl",
        help="Model path. Default is the path to the BASE SVMOH (1, 3) model.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print top 50 entries to terminal. Default: False",
    )

    args = parser.parse_args()
    mid = args.model_tag
    MODEL = args.model_path
    class_label = args.class_label

    label_dictionary = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    class_idx = (
        int(class_label) if len(class_label) == 1 else label_dictionary[class_label]
    )
    m = load_model(MODEL)

    ft = get_top_features_post_reduction(m, class_idx)
    if args.verbose:
        print(ft[0:50])

    ft.to_csv(FEATURES_DIR / f"{mid}_Class_{str(class_idx)}_feature_importance.csv")


if __name__ == "__main__":
    main()
