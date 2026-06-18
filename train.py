import os
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

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


def logistic_regression_one_hot(
    files_to_use, ngram_range=(1, 3), max_iter=1000, C=1.0, models_path="models"
):
    """Trains a logistic regression model using text features for review+target
    and One-Hot Encoding for the aspect category.

    Args:
        files_to_use (list[str]): list of xml files to use for training
        ngram_range (tuple(int)): ngram range to use to the tfidf vectorizer
        max_iter (int): max iterations for logistic regression
        C (float): C parameter of logistic regression
        models_path (str): path (folder name) to store the saved models
    """
    df = concatenate_data(files_to_use)

    # Handle missing values by replacing NaN with an empty string
    df = df.fillna("")

    # Combine text and target together for the TF-IDF feature stream
    df["combined_text"] = df["text"] + " " + df["target"]

    # X must now be a DataFrame containing both columns instead of a single series
    X = df[["combined_text", "category"]]
    y = df["polarity"]

    # Define how each column should be preprocessed
    preprocessor = ColumnTransformer(
        transformers=[
            # 1. Route text column to TF-IDF (expects a 1D column name string)
            (
                "tfidf",
                TfidfVectorizer(ngram_range=ngram_range, analyzer="word"),
                "combined_text",
            ),
            # 2. Route category column to OneHotEncoder (expects a 2D column list)
            # handle_unknown='ignore' ensures test.py won't crash if it sees a new category
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ]
    )

    # Combine preprocessing and model into a single Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=max_iter, C=C, random_state=42)),
        ]
    )

    print("Training model...")
    pipeline.fit(X, y)

    os.makedirs(models_path, exist_ok=True)
    output_file = os.path.join(
        models_path, f"onehot_ngram_{ngram_range}_max_iter_{max_iter}_C_{C}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump(pipeline, f)


def svm_text_features(
    files_to_use, ngram_range=(1, 3), max_iter=1000, C=1.0, models_path="models"
):
    """Trains a Support Vector Machine (LinearSVC) model using the review,
    the target and the category as text features.

    Args:
        files_to_use (list[str]): list of xml files to use for training
        ngram_range (tuple(int)): ngram range to use to the tfidf vectorizer
        max_iter (int): max iterations for the SVM solver
        C (float): Regularization parameter of the SVM
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
            ("classifier", LinearSVC(max_iter=max_iter, C=C, random_state=42)),
        ]
    )

    print("Training SVM model (Text Features)...")
    pipeline.fit(X, y)

    os.makedirs(models_path, exist_ok=True)
    output_file = os.path.join(
        models_path, f"svm_ngram_{ngram_range}_max_iter_{max_iter}_C_{C}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump(pipeline, f)


def svm_one_hot(
    files_to_use, ngram_range=(1, 3), max_iter=1000, C=1.0, models_path="models"
):
    """Trains a Support Vector Machine (LinearSVC) model using text features for review+target
    and One-Hot Encoding for the aspect category.

    Args:
        files_to_use (list[str]): list of xml files to use for training
        ngram_range (tuple(int)): ngram range to use to the tfidf vectorizer
        max_iter (int): max iterations for the SVM solver
        C (float): Regularization parameter of the SVM
        models_path (str): path (folder name) to store the saved models
    """
    df = concatenate_data(files_to_use)

    # Handle missing values by replacing NaN with an empty string
    df = df.fillna("")

    # Combine text and target together for the TF-IDF feature stream
    df["combined_text"] = df["text"] + " " + df["target"]

    # X must now be a DataFrame containing both columns instead of a single series
    X = df[["combined_text", "category"]]
    y = df["polarity"]

    # Define how each column should be preprocessed
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(ngram_range=ngram_range, analyzer="word"),
                "combined_text",
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ]
    )

    # Combine preprocessing and SVM into a single Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LinearSVC(max_iter=max_iter, C=C, random_state=42)),
        ]
    )

    print("Training SVM model (One-Hot)...")
    pipeline.fit(X, y)

    os.makedirs(models_path, exist_ok=True)
    output_file = os.path.join(
        models_path, f"svm_onehot_ngram_{ngram_range}_max_iter_{max_iter}_C_{C}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump(pipeline, f)


# logistic_regression_text_features(
#     [
#         "part1.xml",
#         "part2.xml",
#         "part3.xml",
#         "part4.xml",
#         "part5.xml",
#         "part6.xml",
#         "part7.xml",
#         "part8.xml",
#         "part9.xml",
#         "part10.xml",
#     ]
# )

# logistic_regression_one_hot(
#     [
#         "part1.xml",
#         "part2.xml",
#         "part3.xml",
#         "part4.xml",
#         "part5.xml",
#         "part6.xml",
#         "part7.xml",
#         "part8.xml",
#         "part9.xml",
#         "part10.xml",
#     ]
# )

# svm_text_features(
#     [
#         "part1.xml",
#         "part2.xml",
#         "part3.xml",
#         "part4.xml",
#         "part5.xml",
#         "part6.xml",
#         "part7.xml",
#         "part8.xml",
#         "part9.xml",
#         "part10.xml",
#     ]
# )

# svm_one_hot(
#     [
#         "part1.xml",
#         "part2.xml",
#         "part3.xml",
#         "part4.xml",
#         "part5.xml",
#         "part6.xml",
#         "part7.xml",
#         "part8.xml",
#         "part9.xml",
#         "part10.xml",
#     ]
# )
