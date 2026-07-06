import json
from collections import Counter
import numpy as np
import pandas as pd
import sklearn
from constants import PROJECT_DIR, full_dataset, FINAL_RESULTS
from utils import *

MODEL = r"C:\Users\dioni\ekpa-programming-absa\models\svm_onehot_ngram_(1, 3)_max_iter_1000_C_1-0_reduce_f_False_n_components_1000_.pkl"

m = load_model(MODEL)

def get_top_features_before_reduction(model: str | Path | sklearn.pipeline.Pipeline, target_class=0):
    """Target classes: 0=Negative, 1=Neutral, 2=Positive"""

    if isinstance(model, str | Path):
        model = load_model(model)

    classifier = model.named_steps['classifier']

    if 'preprocessor' in model.named_steps:
        preprocessor = model.named_steps['preprocessor']
    else:
        preprocessor = model.named_steps['tfidf']

    ngrams = preprocessor.get_feature_names_out()
    weights = classifier.coef_[target_class]

    feature_importance = pd.DataFrame({
        "ngrams": ngrams,
        "importance": weights

    })

    top_features = feature_importance.sort_values("importance", ascending=False)

    return top_features


def get_top_features_post_reduction(model: str | Path | sklearn.pipeline.Pipeline, target_class=0):
    """Target classes: 0=Negative, 1=Neutral, 2=Positive"""

    if isinstance(model, str|Path):
        model = load_model(model)

    try:
        red = model.named_steps['reducer']
    except KeyError as e:
        print(f"Model pipeline does not feature a reducer named step! \n{e}")
        return

    classifier = model.named_steps['classifier']

    if 'preprocessor' in model.named_steps:
        preprocessor = model.named_steps['preprocessor']
    else:
        preprocessor = model.named_steps['tfidf']

    orig_weights = np.dot(classifier.coef_, red.components_)
    all_features = preprocessor.get_feature_names_out()

    df_importance = pd.DataFrame({
        "feature": all_features,
        "importance": orig_weights[target_class]
    })

    top_features = df_importance.sort_values(by="importance", key=abs, ascending=False)

    return top_features

ft = get_top_features_before_reduction(m, 0)
print(ft[0:50])

