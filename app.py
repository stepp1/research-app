import json
import logging
import random
import time

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from researcher import Embeddings, extract_paper, visualization_plotly
from researcher.parser.utils import add_to_json, get_authors_str
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
    with st.expander("Want to add a new Paper?"):
        uploaded_file = st.file_uploader("Upload a Paper's PDF", type="pdf")
        if uploaded_file is not None:
            st.write("File uploaded!")

            # parse PDF and extract paper
            result = extract_paper(uploaded_file.name, "papers.json")

            add_to_json(result, current_out_file)

    with open(current_out_file, "r") as f:
        papers = json.load(f)

    sentences = [paper["abstract"] for paper in papers]
    data["sentences"] = sentences
    data["title"] = [paper["title"] for paper in papers]

    st.title("Pipeline Configuration")

    ## preprocessing
    with st.expander("Preprocessing", True):
        sentences_processed = sentences
        use_preproc = st.checkbox(
            "Use Preprocessing Steps to compute embeddings?",
            False,
            help="Use preprocessing steps to compute embeddings? If not, embeddings will be computed on raw data",
        )
        pre = st.multiselect(
            "Preprocessing Steps\n Default steps: Stopwords, Punctuation",
            [
                "Lemmatization",
                "Stemming",
                "Spelling",
                "Clause Separation",
            ],
            help="Preprocessing steps to apply to provided data",
        )
        prep_load()
        sentences_processed = prep_stop(sentences_processed)

        if "Lemmatization" in pre:
            sentences_processed = prep_lemma(sentences_processed)
        if "Stemming" in pre:
            sentences_processed = prep_stem(sentences_processed)
        if "Clause Separation" in pre:
            clause_reg_box = st.text_input(
                "clause sep regex",
                clause_reg,
                help="Regex defining separation of clauses within each sentence/line",
            )
            clause_word_box = st.text_input(
                "clause sep words",
                clause_words,
                help="Words indicating a clause boundary",
            )
            clause_sep = f"{clause_reg}{' | '.join(clause_words)}".replace("] ", "]")
            sentences_processed = prep_clause(sentences_processed)
        data["sentences_processed"] = sentences_processed

    ## embeddings
    with st.expander("Embeddings", True):
        mod = st.selectbox(
            "Embedding Algorithm",
            # ["tf-idf","Hash", "Universal Sentence Encoder"],
            ["Instructor Embeddings", "OpenAI Embeddings", "HuggingFace Embeddings"],
            help="Algorithm used for sentence embeddings, preprocessing steps may be duplicated between the above and the following models. See LangChain's Embeddings Documentation for more information.",
        )

        model_mapping = {
            "Instructor Embeddings": "instructor",
            "OpenAI Embeddings": "openai",
            "HuggingFace Embeddings": "huggingface",
            "tf-idf": "tf-idf",
        }
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
            # emb = model_tfidf(prep, (1,ng))

            ng = st.slider(
                "ngram_range",
                1,
                5,
                help="Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms",
            )
            # emb = model_count(prep, (1,ng))

        if use_preproc:
            embeddings = model.encode(sentences=sentences_processed)
        else:
            embeddings = model.encode(sentences=sentences)
        embeddings = np.array(embeddings)
        logging.info(
            f"Embeddings shape: {embeddings.shape}. Statistics: {embeddings.mean()}, {embeddings.std()}"
        )
        data["embeddings"] = embeddings.tolist()

    ## viz
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
            ["K-Means", "DBSCAN", "Agglomerative Clustering"],
            help="Algorithm used to cluster the embeddings",
        )
        clustering_mapping = {
            "K-Means": "kmeans",
            "DBSCAN": "dbscan",
            "Agglomerative Clustering": "agglomerative",
        }
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

        if mod == "DBSCAN":
            eps = st.slider(
                "eps",
                0.0,
                1.0,
                value=0.5,
                help="The maximum distance between two samples for one to be considered as in the neighborhood of the other",
            )
            min_samples = st.slider(
                "min_samples",
                1,
                10,
                value=4,
                help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point",
            )
            clustering_args["eps"] = eps
            clustering_args["min_samples"] = min_samples

        if mod == "Agglomerative Clustering":
            n_clusters = st.slider(
                "n_clusters", 1, 10, value=4, help="The number of clusters to find"
            )
            affinity = st.selectbox(
                "affinity",
                ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
                help="Metric used to compute the linkage",
            )
            linkage = st.selectbox(
                "linkage",
                ["ward", "complete", "average", "single"],
                help="Which linkage criterion to use",
            )
            clustering_args["n_clusters"] = n_clusters
            clustering_args["affinity"] = affinity
            clustering_args["linkage"] = linkage

        logging.info(
            f"Using {mod} as clustering algorithm. Clustering arguments: {clustering_args}"
        )

        cluster_assignment = model.cluster_assignment(
            embeddings=embeddings, method_name=cluster_method, **clustering_args
        )
        data["cluster_assignment"] = cluster_assignment.tolist()

tab1, tab2 = st.tabs(["📈 Visualization", "🗃 Data"])

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


with tab2:
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
