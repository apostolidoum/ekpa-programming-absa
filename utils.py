import json
import os
import pickle
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path, PurePath

import pandas as pd

from constants import DATA_DIR, full_dataset, nlp


def xml_to_dataframe(xml_filepath):
    """
    Parses an XML part file and converts it into a flattened pandas DataFrame.
    Data Format

    Each row in the CSV files represents one opinion annotation with the following columns:

    | Column        | Description                                      |
    |---------------|--------------------------------------------------|
    | `review_id`   | Unique identifier for the review                 |
    | `sentence_id` | Unique identifier for the sentence               |
    | `text`        | The raw sentence text                            |
    | `target`      | The opinion target (e.g., `"food"`, `"NULL"`)    |
    | `category`    | The aspect category (e.g., `FOOD#QUALITY`)       |
    | `polarity`    | Sentiment polarity (`positive`, `negative`, `neutral`) |

    Sentences with no annotated opinions are still included, with `target`, `category`, and `polarity` set to `None`.
    """
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: The file {xml_filepath} was not found.")
        return None
    except ET.ParseError:
        print(f"Error: Could not parse {xml_filepath}. Ensure it is valid XML.")
        return None

    rows = []

    # Iterate through every <Review> element
    for review in root.findall("Review"):
        review_id = review.attrib.get("rid")

        # Locate the <sentences> container
        sentences_container = review.find("sentences")
        if sentences_container is None:
            continue

        # Iterate through every <sentence> in this review
        for sentence in sentences_container.findall("sentence"):
            sentence_id = sentence.attrib.get("id")

            # Extract sentence text
            text_elem = sentence.find("text")
            sentence_text = text_elem.text if text_elem is not None else ""

            # Locate <Opinions> block
            opinions_container = sentence.find("Opinions")
            opinions = (
                opinions_container.findall("Opinion")
                if opinions_container is not None
                else []
            )

            if not opinions:
                # If there are no opinions, keep the text and fill aspect data with None
                row = {
                    "review_id": review_id,
                    "sentence_id": sentence_id,
                    "text": sentence_text,
                    "target": None,
                    "category": None,
                    "polarity": None,
                }
                rows.append(row)
            else:
                # Create a unique row for each aspect/opinion found in the sentence
                for opinion in opinions:
                    row = {
                        "review_id": review_id,
                        "sentence_id": sentence_id,
                        "text": sentence_text,
                        "target": opinion.attrib.get("target"),
                        "category": opinion.attrib.get("category"),
                        "polarity": opinion.attrib.get("polarity"),
                    }
                    rows.append(row)

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(rows)
    return df


def lemmatize_sentence(docs):
    """
    Takes a sentence or list of strings as input.
    Returns a list of lemmatized sentences.
    """
    if isinstance(docs, str):
        docs = [docs]
    return [
        " ".join([token.lemma_ for token in doc])
        for doc in nlp.pipe(docs, batch_size=256)
    ]


def trim_str_around_target(
    query_str: str, target: str, window=6, token_pattern="default"
):
    """
    Applies context window to string, centered around target word.
    The default token pattern is the same one used by the tf-idf tokenizer, which does not
    preserve aspect labels separated by #. Token pattern 'better' is recommended, kept the default
    one for reproducibility.

    Args:
        query_str (str): query string
        target (str): target word
        window (int): window size on either size
        token_pattern (str): token pattern for tokenization
    """

    patterns = {
        "default": r"(?u)\b[\w\d]\w+\b",
        "better": r"(?u)\b[#\w\d]\w+\b",  # preserves aspect units
    }

    if token_pattern in patterns:
        token_pattern = patterns[token_pattern]

    query_str = query_str.lower()
    target = target.lower()
    tokens = re.findall(token_pattern, query_str)  # use default pattern used by tfidf

    if target in query_str:
        lwin = rwin = window
        target_tokens = re.findall(token_pattern, target)

        if (len(target_tokens) > 1) or (target_tokens[0] != target):
            if target_tokens[0]:
                target = target_tokens[0]  # focus on first word of the target
                rwin += len(target_tokens) - 1
            else:
                target = target_tokens[1]
                rwin += len(target_tokens) - 2
                lwin += 1

        try:
            target_idx = tokens.index(target)
        except:
            print(tokens, target, target_tokens)

        start_idx = max(target_idx - lwin, 0)
        end_idx = min(target_idx + rwin, len(tokens))

        return " ".join(tokens[start_idx:end_idx])

    return " ".join(tokens)


def save_csv(df, filename, output_dir=DATA_DIR):
    """Save a dataframe to csv at the output directory under the specified filename"""

    df.to_csv(f"{output_dir}/{filename}.csv", index=False, encoding="utf-8")
    print(f"Dataframe saved to {output_dir}/{filename}.csv")


def concatenate_data(files_to_use):
    """Takes a list of xml filenames and returns a dataframe with the concatenation of the data"""

    dataframes = []
    for file in files_to_use:
        df = xml_to_dataframe(os.path.join("data", file))
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def get_feature_dimensionality(clf: str | Path):
    """Returns tuple containing (n_classes, n_features).
    Accepts a path as input."""

    if "models" in PurePath(clf).parts:
        clf = load_model(clf)
    else:
        clf = load_model(Path("models", clf))

    return clf["classifier"].coef_.shape


def split_features_from_target(
    df: pd.DataFrame, key="one-hot", lngrams=False, target_context_window=False
):
    """Preprocesses initial dataframe according to model specifications and separates X from y,
    ignoring all redundant columns.

    The expected dataframe input is of the structure:

    | Column        | Description                                      |
    |---------------|--------------------------------------------------|
    | `review_id`   | Unique identifier for the review                 |
    | `sentence_id` | Unique identifier for the sentence               |
    | `text`        | The raw sentence text                            |
    | `target`      | The opinion target (e.g., `"food"`, `"NULL"`)    |
    | `category`    | The aspect category (e.g., `FOOD#QUALITY`)       |
    | `polarity`    | Sentiment polarity (`positive`, `negative`, `neutral`) |

    Args:
        df (pd.DataFrame): Dataframe containing data
        key (str): aspect label encoding
        lngrams (bool): use lemma n-grams
        target_context_window (bool): use context window size around target words
    """

    feature_cols = ["text", "target", "category"]

    df = df.copy()
    df = df.dropna()

    X = df[feature_cols]
    y = df["polarity"]

    if target_context_window:
        combined_docs = X.apply(
            lambda row: trim_str_around_target(row["text"], row["target"]), axis=1
        )
    else:
        combined_docs = X["text"].str.cat(X["target"], sep=" ")

    if key == "one-hot":
        X["combined_text"] = (
            lemmatize_sentence(combined_docs.tolist()) if lngrams else combined_docs
        )
        X = X[
            ["combined_text", "category"]
        ]  # returns a 2-column DataFrame containing all rows
    else:
        X["text"] = (
            lemmatize_sentence(combined_docs.tolist()) if lngrams else combined_docs
        )
        X = X["text"].str.cat(X["category"], sep=" ")

    print(X.head())
    return X, y


def load_model(path):
    """Load model from input address"""
    with open(path, "rb") as f:
        clf = pickle.load(f)

    return clf


def get_portions(stuff: list | pd.Series):
    """Convert raw amounts to percentages."""
    counts = Counter(stuff)
    total = counts.total()
    portions = {k: (v, (v / total) * 100) for k, v in counts.items()}

    return portions


def compute_dataset_statistics(portions=False):
    """Exports a full statistical report on the xml dataset to PROJECT_DIR/stats.json.

    Args:
        portions (bool, optional): Whether to include portion percentages of each category. Defaults to False.
    """
    data: pd.DataFrame = concatenate_data(full_dataset)
    data = data.dropna()

    polarities = data["polarity"]
    sentences = data["sentence_id"]
    composite_categories = data["category"]
    category_units = [com for cat in composite_categories for com in cat.split("#")]

    unit_count = get_portions(category_units) if portions else Counter(category_units)
    cat_count = (
        get_portions(composite_categories)
        if portions
        else Counter(composite_categories)
    )
    pol_count = get_portions(polarities) if portions else Counter(polarities)

    n_reviews = data.shape[0]
    unique_sentences = set(sentences.tolist())

    stats = {
        "Number of Reviews": n_reviews,
        "Number of Unique Sentences": len(unique_sentences),
        "Polarity Distribution": pol_count,
        "Composite Categories": cat_count,
        "Category Unit Count": unit_count,
    }

    address = DATA_DIR / "stats.json"
    json.dump(stats, open(address, "w"))
    print("Statistics saved to " + str(address))
    return address


def get_model_name_from_path(path: Path):
    """Extract model name from path."""
    model_name = path.stem
    return model_name


if __name__ == "__main__":
    # go over all the parts in the data folder and save them to csv
    compute_dataset_statistics()
    for i in range(1, 11):
        df = xml_to_dataframe(f"{DATA_DIR}/part{i}.xml")
        save_csv(df, f"part{i}", DATA_DIR)
