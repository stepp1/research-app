import random
import re
import string

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

random.seed(42)


@st.cache_resource
def prep_load():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def prep_lower(in_text):
    return [i.lower() for i in in_text]


def prep_punct(in_text):
    return [i.translate(str.maketrans("", "", string.punctuation)) for i in in_text]


def prep_stop(in_text):
    t = []
    final_stops = stopwords.words() + [
        "present",
        "paper",
        "abstract",
        "score",
        "review",
        "art",
        "part",
        "state",
        "state-of-the-art",
        "matching",
        # "self-training",
        "neural",
    ]
    for i in in_text:
        cleaned = []
        for j in word_tokenize(i.lower()):
            if "the art" in j:
                print(j)
            if "state of" in j:
                print(j)
            if j not in final_stops:
                cleaned.append(j)

        t.append(" ".join(cleaned))
    return t


def prep_lemma(in_text):
    t = []
    for i in in_text:
        t.append(" ".join([WordNetLemmatizer().lemmatize(k) for k in word_tokenize(i)]))
    return t


def prep_stem(in_text):
    t = []
    for i in in_text:
        t.append(" ".join([PorterStemmer().stem(k) for k in word_tokenize(i)]))
    return t


clause_reg = "[\.\!\\\/\|,\?\;\:_\-=+]"
clause_words = [
    "and",
    "about",
    "but",
    "so",
    "because",
    "since",
    "though",
    "although",
    "unless",
    "however",
    "until",
]
clause_sep = f"{clause_reg}{' | '.join(clause_words)}".replace("] ", "]")


def prep_clause(in_text):
    t = []
    for i in in_text:
        for j in re.split(clause_sep, i, flags=re.IGNORECASE):
            if j != "":
                t.append(str.strip(j))
    return t


def prep_ex(in_text, func):
    out = pd.DataFrame.from_dict({"Before": in_text, "After": func(in_text)})
    return out
