import json
import logging

import numpy as np
import streamlit as st

from researcher import Embeddings, extract_paper
from researcher.parser.utils import add_to_json
from researcher.representations.preprocessing import *
from researcher.representations.stopwords import prep_load

clustering_mapping = {
    "K-Means": "kmeans",
    "DBSCAN": "dbscan",
    "Agglomerative Clustering": "agglomerative",
}
model_mapping = {
    "Instructor Embeddings": "instructor",
    "OpenAI Embeddings": "openai",
    "HuggingFace Embeddings": "huggingface",
    "tf-idf": "tf-idf",
}


def upload_sidebar(current_out_file):
    # with st.expander("Want to add a new Paper?"):
    #     uploaded_file = st.file_uploader("Upload a Paper's PDF", type="pdf")
    #     if uploaded_file is not None:
    #         st.write("File uploaded!")

    #         # parse PDF and extract paper
    #         result = extract_paper(uploaded_file.name, "papers.json")

    #         add_to_json(result, current_out_file)

    with open(current_out_file, "r") as f:
        papers = json.load(f)

    return papers


def preprocess_sidebar(sentences, data):
    ## preprocessing
    with st.expander("Preprocessing", True):
        data["sentences_processed"] = sentences
        use_preproc = st.checkbox(
            "Use Preprocessing Steps to compute embeddings?",
            False,
            help="Use preprocessing steps to compute embeddings? If not, embeddings will be computed on raw data",
        )
        pre = st.multiselect(
            "Preprocessing Steps\n Default steps: Stopwords, Punctuation, Lowercase, Remove Digits, Remove Whitespace",
            [
                "Lemmatization",
                "Stemming",
                "Spelling",
                "Clause Separation",
            ],
            help="Preprocessing steps to apply to provided data",
        )
        prep_load()

        data["sentences_processed"] = clean(data["sentences_processed"])

        if "Lemmatization" in pre:
            data["sentences_processed"] = build_lemma(data["sentences_processed"])
        if "Stemming" in pre:
            data["sentences_processed"] = build_stem(data["sentences_processed"])
        if "Clause Separation" in pre:
            clause_word_box = st.text_input(
                "clause sep words",
                DEFAULT_CLAUSE_WORDS,
                help="Words indicating a clause boundary",
            )
            sentences_processed = build_clause(
                sentences_processed, custom_clause_word=clause_word_box
            )

    return use_preproc


def embeddings_sidebar(sentences, sentences_processed, use_preproc, data):
    ## embeddings
    with st.expander("Embeddings", True):
        mod = st.selectbox(
            "Embedding Algorithm",
            # TODO: add tf-idf, Universal Sentence Encoder
            ["Instructor Embeddings", "OpenAI Embeddings", "HuggingFace Embeddings"],
            help="Algorithm used for sentence embeddings, preprocessing steps may be duplicated between the above and the following models. See LangChain's Embeddings Documentation for more information.",
        )

        model_name = model_mapping[mod]
        logging.info(f"Using {mod} as embedding algorithm")

        if model_name == "openai":
            openai_key = st.text_input(
                "OpenAI API Key",
                "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxx",
                help="API Key for OpenAI's API",
            )
            model = Embeddings(model_name, key=openai_key)
        elif model_name in ["instructor", "huggingface"]:
            model = Embeddings(model_name)

        elif mod == "tf-idf":
            ng = st.slider(
                "ngram_range",
                1,
                5,
                help="Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms",
            )
            # TODO: emb = model_tfidf(prep, (1,ng))

            ng = st.slider(
                "ngram_range",
                1,
                5,
                help="Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms",
            )
            # TODO: emb = model_count(prep, (1,ng))

        if use_preproc:
            embeddings = model.encode(sentences=sentences_processed)
        else:
            embeddings = model.encode(sentences=sentences)
        embeddings = np.array(embeddings)
        logging.info(
            f"Embeddings shape: {embeddings.shape}. Statistics: {embeddings.mean()}, {embeddings.std()}"
        )
        data["embeddings"] = embeddings.tolist()
    return model, model_name


def visualization_sidebar(model, embeddings=None, data=None):
    # must provide embeddings or model.embeddings
    if embeddings is None:
        embeddings = model.embeddings

    # visualization: dimensionality reduction and clustering
    with st.expander("Visualization", True):
        mod = st.selectbox(
            "Dimensionality Reduction",
            ["PCA", "t-SNE", "UMAP"],
            help="Algorithm used to reduce the dimensionality of the embeddings",
        )

        decompose_method = mod.lower().replace("-", "")
        logging.info(f"Using {mod} as dimensionality reduction algorithm")

        mod = st.selectbox(
            "Clustering Algorithm",
            ["K-Means", "Agglomerative Clustering"],
            help="Algorithm used to cluster the embeddings",
        )

        cluster_method = clustering_mapping[mod]

        clustering_args = {}

        if mod == "K-Means":
            n_clusters = st.slider(
                "n_clusters",
                1,
                10,
                value=4,
                help="Number of clusters to use for k-means clustering",
            )
            clustering_args["n_clusters"] = n_clusters

        if mod == "Agglomerative Clustering":
            n_clusters = st.slider(
                "n_clusters", 1, 10, value=4, help="The number of clusters to find"
            )
            metric = st.selectbox(
                "metric",
                ["euclidean", "l1", "l2", "manhattan", "cosine"],
                help="Metric used to compute the linkage",
            )
            linkage = st.selectbox(
                "linkage",
                ["ward", "complete", "average", "single"],
                help="Which linkage criterion to use",
            )
            clustering_args["n_clusters"] = n_clusters
            clustering_args["metric"] = metric
            clustering_args["linkage"] = linkage

        logging.info(
            f"Using {mod} as clustering algorithm. Clustering arguments: {clustering_args}"
        )

        cluster_assignment = model.cluster(
            embeddings=embeddings, method_name=cluster_method, **clustering_args
        )
        data["cluster_assignment"] = cluster_assignment.tolist()

    return cluster_method, decompose_method


def load_sidebar(data, current_out_file):
    papers = upload_sidebar(current_out_file)

    sentences = [paper["abstract"] for paper in papers]
    data["sentences"] = sentences
    data["title"] = [paper["title"] for paper in papers]

    st.title("Pipeline Configuration")

    use_preproc = preprocess_sidebar(sentences, data)
    model, model_name = embeddings_sidebar(
        sentences=data["sentences"],
        sentences_processed=data["sentences_processed"],
        use_preproc=use_preproc,
        data=data,
    )
    cluster_method, decompose_method = visualization_sidebar(model=model, data=data)

    data.papers = papers
    data.use_preproc = use_preproc
    data.cluster_method = cluster_method
    data.decompose_method = decompose_method
    data.model_name = model_name
    return data, model
