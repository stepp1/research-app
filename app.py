import json
import logging
import random
import time

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from app.sidebar import (
    embeddings_sidebar,
    model_mapping,
    preprocess_sidebar,
    upload_sidebar,
    visualization_sidebar,
)
from researcher import visualization_plotly
from researcher.parser.utils import get_authors_str
from researcher.preprocessing import *

st.set_page_config(layout="wide")
random.seed(42)
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

start_time = time.time()
st.title("Machine Learning Research: A Paper based Approach")

data_source = st.selectbox(
    "Data Source (Currently only Abstracts are supported)",
    ["Abstract"],  # ["Title", "Abstract", "Full Text"],
    help="Choose the data source for the pipeline",
    format_func=lambda x: x.title(),
)
if data_source == "Abstract":
    current_out_file = "researcher/out/result.json"
else:
    current_out_file = "researcher/out/result.json"

logging.info(f"Using {data_source} as data source. Loaded from {current_out_file}")


data = pd.DataFrame(
    columns=[
        "sentences",
        "sentences_processed",
        "title",
        "embeddings",
        "cluster_assignment",
        "top_words",
    ]
)


with st.sidebar:
    papers = upload_sidebar(current_out_file)

    sentences = [paper["abstract"] for paper in papers]
    data["sentences"] = sentences
    data["title"] = [paper["title"] for paper in papers]

    st.title("Pipeline Configuration")

    sentences_processed, use_preproc = preprocess_sidebar(sentences, data)
    embeddings, model, model_name = embeddings_sidebar(
        sentences, sentences_processed, use_preproc, data
    )
    cluster_assignment, cluster_method, decompose_method = visualization_sidebar(
        model, embeddings, data
    )


font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 20px;
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Visualization", "PaperChat", "🗃 Papers"])

with tab1:
    st.subheader("Paper Explorer")

    logging.info("Clustering embeddings")

    ###### TODO: (step) clean up this code
    # group sentences by cluster
    cluster_sentences = {}
    for i, cluster in enumerate(cluster_assignment):
        if cluster not in cluster_sentences:
            cluster_sentences[cluster] = []
        cluster_sentences[cluster].append(sentences_processed[i])

    cluster_tops = {}
    for cluster_k, cluster_values in cluster_sentences.items():
        # compute tf-idf
        vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_features=1000)
        tfidf = vectorizer.fit_transform(cluster_values)

        # obtain the top 10 words
        top_words = np.array(tfidf.sum(axis=0).tolist()[0]).argsort()[-4:][::-1]

        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words]

        cluster_tops[cluster_k] = top_words

    top_words = []
    for i, cluster in enumerate(cluster_assignment):
        top_words.append(str(cluster_tops[cluster]))

    data["top_words"] = top_words
    ######

    data["color"] = top_words
    data["label"] = data["title"]

    mod_model_name = {v: k for k, v in model_mapping.items()}[model_name]
    title_add = f" - {mod_model_name}"
    fig = visualization_plotly(
        data=data, decompose_method=decompose_method, title_add=title_add
    )

    logging.info("Plotting visualization")
    st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.subheader("Current Papers")

    # 3 columns
    col1, col2, col3 = st.columns(3)

    for i, paper in enumerate(papers):
        if i % 3 == 0:
            current_col = col1
        elif i % 3 == 1:
            current_col = col2
        else:
            current_col = col3

        with current_col.expander(f"**{paper['title']}**", False):
            st.markdown(f"* Authors: {get_authors_str(paper['authors'])}")
            st.markdown(f"* Abstract: {paper['abstract']}")
            st.markdown(f"* Link: {paper['url']}")
            st.markdown(" ")

logging.info(
    "App loaded. Loading time: {:.2f} seconds".format(time.time() - start_time)
)
