from collections import Counter

import pandas as pd

from constants import full_dataset
from utils import concatenate_data, get_portions


def filter_by_component(df, component):
    return df[df["category"].str.contains(component, na=False)]


def get_category_components(categories):
    return [com for cat in categories for com in cat.split("#")]


data: pd.DataFrame = concatenate_data(full_dataset)
pruned_data = data.dropna()

polarities = pruned_data["polarity"]
reviews = pruned_data["review_id"]
texts = data["sentence_id"]
composite_categories = pruned_data["category"]

category_units = [
    com
    for cat in composite_categories
    if isinstance(cat, str)
    for com in cat.split("#")
]

portions = False
unit_count = get_portions(category_units) if portions else Counter(category_units)
cat_count = (
    get_portions(composite_categories) if portions else Counter(composite_categories)
)
pol_count = get_portions(polarities) if portions else Counter(polarities)

n_entries = pruned_data.shape[0]
n_reviews = len(set(reviews.tolist()))
unique_sentences = set(texts.tolist())
___after_drop_na = set(pruned_data.dropna()["sentence_id"].tolist())

stats = {
    "Number of Entries": n_entries,
    "Number of Reviews": n_reviews,
    "Number of Unique Sentences": len(unique_sentences),
    "Number of Unique Sentences after pruning": len(___after_drop_na),
    "Polarity Distribution": pol_count,
    "Composite Categories": cat_count,
    "Category Unit Count": unit_count,
}
print(stats)

# data = pd.read_csv('predictions.csv')
#
# print(data)
#
# wrong_responses = data[data['truth']!=data['preds']]
# _all_categories = data['category']
# all_categories = Counter(data['category'])
# all_components = get_category_components(_all_categories)
#
# _wrong_categories = wrong_responses['category']
# components = get_category_components(_wrong_categories)
#
# wrong_categories = Counter(_wrong_categories)
# wrong_category_components = Counter(components)
#
#
# print(all_categories)
# print(wrong_categories)
#
# print(Counter(all_components))
# print(wrong_category_components)
#
# print(filter_by_component(data, 'FOOD'))
