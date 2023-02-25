import nltk
import streamlit as st


@st.cache_resource
def prep_load():
    nltk.download("punkt")
    try:
        # If not present, download NLTK stopwords.
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


from nltk.corpus import stopwords as nltk_en_stopwords

custom_stopwords = set(
    [
        "test",
        "pseudo",
        "using",
        "methods",
        "show",
        "training",
        "data",
        ",",
        ".",
        "models",
        # "present",
        # "paper",
        # "abstract",
        # "score",
        # "review",
        # "art",
        # "part",
        "state",
        "state-of-the-art",
        # "matching",
        # # "self-training",
        # "based"
        # "’",
        # "“",
        # "”",
        # "'",
        # "neural",
    ]
)
NLTK_EN = set(nltk_en_stopwords.words("english")).union(custom_stopwords)
DEFAULT = NLTK_EN.union(custom_stopwords)
