import re

import spacy
from spacy.cli import download
from spacy.util import is_package

model_name = "en_core_web_sm"

# 1. Instantly check if the model exists; download ONLY if missing
if not is_package(model_name):
    print(f"Model '{model_name}' not found. Downloading...")
    download(model_name)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def lemmatize_sentence(docs):
    if isinstance(docs, str):
        docs = [docs]
    return [" ".join([token.lemma_ for token in doc])
        for doc in nlp.pipe(docs, batch_size=256)]

def trim_str_around_target(query_str:str, target:str, window=6, token_pattern=r"(?u)\b[\w\d]\w+\b"):
    query_str = query_str.lower()
    target = target.lower()
    tokens = re.findall(token_pattern, query_str) # use default pattern used by tfidf

    if target in query_str:
        lwin = rwin = window
        target_tokens = re.findall(token_pattern, target)

        if (len(target_tokens) > 1) or (target_tokens[0] != target):
            if target_tokens[0]:
                target = target_tokens[0] #focus on first word of the target
                rwin += len(target_tokens) - 1
            else:
                target = target_tokens[1]
                rwin += len(target_tokens) - 2
                lwin += 1

        try:
            target_idx = tokens.index(target)
        except:
            print(tokens, target, target_tokens)

        start_idx = max(target_idx-lwin, 0)
        end_idx = min(target_idx+rwin, len(tokens))

        return " ".join(tokens[start_idx:end_idx])

    return " ".join(tokens)


print(nlp.tokenizer("This was some of the best well-cooked meat I have ever had in my life!"))







