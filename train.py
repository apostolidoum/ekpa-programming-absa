import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import concatenate_data


def logistic_regression_text_features(
    files_to_use, ngram_range=(1, 3), max_iter=1000, C=1.0, models_path="models"
):
    """Trains a logistic regression model using the review, the target and the category as text features.

    Args:
        files_to_use (list[str]): list of xml files to use for training
        ngram_range (tuple(int)): ngram range to use to the tfidf vectorizer
        max_iter (int): max iterations for logistic regression
        C (float): C parameter of logistic regression
        models_path (str): path (folder name) to store the saved models
    """
    df = concatenate_data(files_to_use)

    # Replace all NaN values across the DataFrame with an empty string
    df = df.fillna("")

    X = df["text"] + " " + df["target"] + " " + df["category"]
    y = df["polarity"]

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, analyzer="word")),
            ("classifier", LogisticRegression(max_iter=max_iter, C=C, random_state=42)),
        ]
    )

    print("Training model...")
    pipeline.fit(X, y)

    os.makedirs(models_path, exist_ok=True)
    output_file = os.path.join(
        models_path, f"ngram_{ngram_range}_max_iter_{max_iter}_C_{C}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump(pipeline, f)


logistic_regression_text_features(["part1.xml", "part2.xml"])
