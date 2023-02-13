import json
import random
from researcher.parser.utils import get_authors_str

import streamlit as st

st.set_page_config(layout="wide")
random.seed(42)

st.title("Machine Learning Research: A Paper based Approach")

data_source = st.selectbox(
    "Data Source",
    ["Title", "Abstract", "Full Text"],
    help = "Choose the data source for the pipeline",
    format_func=lambda x: x.title()
)

with st.sidebar:
    with st.expander("Want to add a new Paper?"):
        uploaded_file = st.file_uploader("Upload a Paper's PDF", type="pdf")
    
    st.title("Pipeline Configuration")
    
    ## preprocessing
    with st.expander("Text Preprocessing", False):
        prep = data_source
        pre = st.multiselect(
            "Preprocessing Steps", 
            ["Lowercase","Punctuation","Stopwords","Lemmatization","Stemming","Spelling","Clause Separation"], 
            help = "Preprocessing steps to apply to provided data"
        )
        # prep_load()

        # if "Lowercase" in pre:
        #     prep = prep_lower(prep)
        # if "Punctuation" in pre:
        #     prep = prep_punct(prep)
        # if "Stopwords" in pre:
        #     prep = prep_stop(prep)
        # if "Lemmatization" in pre:
        #     prep = prep_lemma(prep)
        # if "Stemming" in pre:
        #     prep = prep_stem(prep)
        # if "Spelling" in pre:
        #     prep = prep_spell(prep)
        # if "Clause Separation" in pre:
        #     clause_reg_box = st.text_input("clause sep regex", clause_reg, help = "Regex defining separation of clauses within each sentence/line")
        #     clause_word_box = st.text_input("clause sep words", clause_words, help = "Words indicating a clause boundary")
        #     clause_sep = f"{clause_reg}{' | '.join(clause_words)}".replace("] ", "]")
        #     prep = prep_clause(prep)

    ## embeddings
    with st.expander("Sentence Embeddings", False):
        mod = st.selectbox(
            "Embedding Algorithm", 
            ["tf-idf","Hash","Count","SentenceTransformers Model","Universal Sentence Encoder"], 
            help = "Algorithm used for sentence embeddings, preprocessing steps may be duplicated between the abova and the following models"
        )

        if mod == "tf-idf":
            ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
            # emb = model_tfidf(prep, (1,ng))
        if mod == "Hash":
            ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
            # emb = model_hash(prep, (1,ng))
        if mod == "Count":
            ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
            # emb = model_count(prep, (1,ng))
        # if mod == "SentenceTransformers Model":
            # st_mod = st.selectbox("st model selection", st_available_models, help = "Pretrained models available through the SetnenceTransformers library and HuggingFace.co")
            # emb = model_snt(prep, st_mod)
        if mod == "Universal Sentence Encoder":
            # emb = model_use(prep)
            ...


    
# add two tabs
tab1, tab2 = st.tabs(["📈 Visualization", "🗃 Data"])

with tab1:
    st.subheader("Paper Explorer")

with tab2:
    st.subheader("Current Papers")

    with open("out/result.json", "r") as f:
        papers = json.load(f)
    for paper in papers:
        exp = st.expander(paper["title"])
        authors = get_authors_str(paper["authors"])
        exp.markdown(f"- **Authors:** {authors}")
        exp.markdown(f"- **URL:** {paper['url']}")
