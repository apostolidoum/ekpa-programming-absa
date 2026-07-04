import pandas as pd
from collections import Counter

def filter_by_component(df, component):
    return df[df['category'].str.contains(component, na=False)]

def get_category_components(categories):
    return [com for cat in categories for com in cat.split('#')]

data = pd.read_csv('predictions.csv')

print(data)

wrong_responses = data[data['truth']!=data['preds']]
_all_categories = data['category']
all_categories = Counter(data['category'])
all_components = get_category_components(_all_categories)

_wrong_categories = wrong_responses['category']
components = get_category_components(_wrong_categories)

wrong_categories = Counter(_wrong_categories)
wrong_category_components = Counter(components)


print(all_categories)
print(wrong_categories)

print(Counter(all_components))
print(wrong_category_components)

print(filter_by_component(data, 'FOOD'))