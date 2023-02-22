import logging
import random
import time

import streamlit as st

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
    current_out_file = "researcher/data/dataset.json"
else:
    current_out_file = "researcher/data/dataset.json"

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


def sidebar(current_out_file):
    with st.sidebar:
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

        data.cluster_method = cluster_method
        data.decompose_method = decompose_method

    return papers, model, model_name, data


def main():
    papers, model, model_name, data = sidebar(current_out_file=current_out_file)

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

        # get top words
        data = model.get_top_words(sentences=data["sentences_processed"], data=data)

        data["color"] = data["top_words"]
        data["label"] = data["title"]

        mod_model_name = {v: k for k, v in model_mapping.items()}[model_name]
        title_add = f" - {mod_model_name}"
        fig = visualization_plotly(
            data=data, decompose_method=data.decompose_method, title_add=title_add
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

    logging.info(f"App loaded. Loading time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
