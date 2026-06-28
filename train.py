import os
import pickle
import argparse

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD

from utils import concatenate_data, split_features_from_target


def logistic_regression_text_features(
    files_to_use, ngram_range=(1, 3), max_iter=1000, C=1.0, models_path="models", reduce_f=False, n_components=1000
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

    X, y = split_features_from_target(df, 'text_f')

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, analyzer="word")),
            ("reducer", TruncatedSVD(n_components=n_components, random_state=42)),
            ("classifier", LogisticRegression(max_iter=max_iter, C=C, random_state=42)),
        ] if reduce_f else [
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, analyzer="word")),
            ("classifier", LogisticRegression(max_iter=max_iter, C=C, random_state=42)),]
    )

    print("Training model...")
    pipeline.fit(X, y)

    os.makedirs(models_path, exist_ok=True)
    output_file = os.path.join(
        models_path, f"ngram_{ngram_range}_max_iter_{max_iter}_C_{C}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump(pipeline, f)

    return output_file


def logistic_regression_one_hot(
    files_to_use, ngram_range=(1, 3), max_iter=1000, C=1.0, models_path="models", reduce_f=False, n_components=1000
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

    X, y = split_features_from_target(df, 'one-hot')

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
            ("reducer", TruncatedSVD(n_components=n_components, random_state=42)),
            ("classifier", LogisticRegression(max_iter=max_iter, C=C, random_state=42)),
        ] if reduce_f else [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=max_iter, C=C, random_state=42)),]
    )

    print("Training model...")
    pipeline.fit(X, y)

    os.makedirs(models_path, exist_ok=True)
    output_file = os.path.join(
        models_path, f"onehot_ngram_{ngram_range}_max_iter_{max_iter}_C_{C}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump(pipeline, f)

    return output_file


def svm_text_features(
    files_to_use, ngram_range=(1, 3), max_iter=1000, C=1.0, models_path="models", reduce_f=False, n_components=1000
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

    X, y = split_features_from_target(df, 'text_f')

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, analyzer="word")),
            ("reducer", TruncatedSVD(n_components=n_components, random_state=42)),
            ("classifier", LinearSVC(max_iter=max_iter, C=C, random_state=42)),
        ] if reduce_f else [
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, analyzer="word")),
            ("classifier", LinearSVC(max_iter=max_iter, C=C, random_state=42)),]
    )

    print("Training SVM model (Text Features)...")
    pipeline.fit(X, y)

    os.makedirs(models_path, exist_ok=True)
    output_file = os.path.join(
        models_path, f"svm_ngram_{ngram_range}_max_iter_{max_iter}_C_{C}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump(pipeline, f)

    return output_file

def svm_one_hot(
    files_to_use, ngram_range=(1, 3), max_iter=1000, C=1.0, models_path="models", reduce_f=False, n_components=1000
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

    X, y = split_features_from_target(df, 'one-hot')

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
            ("reducer", TruncatedSVD(n_components=n_components, random_state=42)),
            ("classifier", LinearSVC(max_iter=max_iter, C=C, random_state=42)),
        ] if reduce_f else [
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
        print(f"svm_onehot_ngram_{ngram_range}_max_iter_{max_iter}_C_{C}.pkl saved to {models_path}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Run logistic regression on text features'
    )

    # Positional argument for XML files
    parser.add_argument(
        'test_files',
        nargs='+',
        help='XML file paths (e.g., part1.xml part2.xml ...) that will be EXCLUDED from training'
    )

    parser.add_argument(
        '--model',
        choices=['svm-tf', 'svm-oh', 'logr-tf', 'logr-oh'],
        default='logr-tf',
        help='Model type to use (default: logr-tf)'
    )

    # Optional arguments
    parser.add_argument(
        '--ngram-range',
        type=int,
        nargs=2,
        default=[1, 3],
        metavar=('MIN', 'MAX'),
        help='N-gram range (default: 1 3)'
    )

    parser.add_argument(
        '--max-iter',
        type=int,
        default=1000,
        help='Maximum iterations (default: 1000)'
    )

    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='Regularization parameter (default: 1.0)'
    )

    parser.add_argument(
        '--models-path',
        default='models',
        help='Path to save models (default: models)'
    )

    args = parser.parse_args()

    full_file_list = [
         "part1.xml",
         "part2.xml",
         "part3.xml",
         "part4.xml",
         "part5.xml",
         "part6.xml",
         "part7.xml",
         "part8.xml",
         "part9.xml",
         "part10.xml",
     ]

    common_params = {
        'files_to_use': [f for f in full_file_list if f not in args.test_files],
        'ngram_range': tuple(args.ngram_range),
        'max_iter': args.max_iter,
        'C': args.C,
        'models_path': args.models_path
    }

    # Choose model based on argument ['svm-tf', 'svm-oh', 'logr-tf', 'logr-oh']
    if args.model == 'svm-tf':
        _ = svm_text_features(**common_params)
    elif args.model == 'svm-oh':
        _ = svm_one_hot(**common_params)
    elif args.model == 'logr-tf':
        _ = logistic_regression_text_features(**common_params)
    elif args.model == 'logr-oh':
        _ = logistic_regression_one_hot(**common_params)



if __name__ == '__main__':
    main()


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
