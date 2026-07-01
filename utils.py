import os
import xml.etree.ElementTree as ET
import pickle

import pandas as pd


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


def save_csv(df, filename, output_dir):
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


def has_preprocessor(clf):
    return "preprocessor" in clf.named_steps


def get_feature_dimensionality(clf:str):

    if "models/" in clf:
        clf = load_model(clf)
    else:
        clf = load_model(os.path.join("models", clf))

    return clf["classifier"].coef_.shape


def split_features_from_target(df: pd.DataFrame, key="one-hot"):
    # Explicitly pull your feature columns and your label column by name
    feature_cols = ["text", "target", "category"]

    df = df.dropna()

    X = df[feature_cols]
    y = df["polarity"]  # Replace "sentiment" with your actual exact target column name

    if key == "one-hot":
        X["combined_text"] = X["text"] + " " + X["target"]
        # Explicitly returns a 2-column DataFrame containing all rows
        X = X[["combined_text", "category"]]
    else:
        X = X["text"] + " " + X["target"] + " " + X["category"]

    return X, y


def load_model(path):

    with open(path, "rb") as f:
        clf = pickle.load(f)

    return clf


# def prepare_features(df: pd.DataFrame, key='one-hot'):
#     # Work on a copy to prevent SettingWithCopyWarning in Pandas
#     df = df.copy()
#
#     if key == 'one-hot':
#         df["combined_text"] = df["text"] + " " + df["target"]
#         # Explicitly returns a 2-column DataFrame containing all rows
#         X = df[["combined_text", "category"]]
#     else:
#         X = df["text"] + " " + df["target"] + " " + df["category"]
#
#     return X

if __name__ == "__main__":
    # go over all the parts in the data folder and save them to csv
    data_path = "data"
    for i in range(1, 11):
        df = xml_to_dataframe(f"{data_path}/part{i}.xml")
        save_csv(df, f"part{i}", data_path)
