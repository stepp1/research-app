import logging
import random
import time

import pandas as pd
import streamlit as st

from app.pages.sidebars.explorer import load_sidebar, model_mapping
from researcher.parser.utils import get_authors_str
from researcher.viz import scatterplot

random.seed(42)

start_time = time.time()

current_out_file = "researcher/data/dataset.json"

st.header("Paper Explorer")

logging.info(f"Loaded {current_out_file}")


def paper_explorer(data, model, model_mapping, model_name):
    logging.info("Clustering embeddings")

    # get top words
    data["Top Words per Cluster"] = model.get_top_words(
        sentences=data["sentences_processed"], data=data
    )

    mod_model_name = {v: k for k, v in model_mapping.items()}[model_name]
    title_add = f" - {mod_model_name}"
    fig = scatterplot(
        data_frame=data,
        decompose_method=data.decompose_method,
        color="Top Words per Cluster",
        hover_name="title",
        title_add=title_add,
        return_figure=True,
    )

    logging.info("Plotting visualization")
    st.plotly_chart(fig, use_container_width=True)

    return data


def main():
    data = pd.DataFrame(
        columns=[
            "sentences",
            "sentences_processed",
            "title",
            "embeddings",
            "cluster_assignment",
            "topwords",
        ]
    )
    with st.sidebar:
        data, model = load_sidebar(data, current_out_file)
    # data.papers = papers
    # data.use_preproc = use_preproc
    # data.cluster_method = cluster_method
    # data.decompose_method = decompose_method
    # data.model_name = model_name

    font_css = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
    }
    </style>
    """

    st.write(font_css, unsafe_allow_html=True)

    # tab1 = st.tabs(["📈 Visualization", "🦜PaperChat", "📄 Papers"])

    # with tab1:
    data = paper_explorer(data, model, model_mapping, data.model_name)

    st.subheader("Current Papers")

    df_papers = pd.DataFrame(data.papers)
    logging.info(f"{df_papers.columns}")
    df_papers["authors"] = df_papers["authors"].apply(get_authors_str)
    df_papers[["Title", "Authors", "Abstract", "URL"]] = df_papers[
        ["title", "authors", "abstract", "url"]
    ]
    st.dataframe(
        df_papers[["Title", "Authors", "Abstract", "URL"]], width=1000, height=500
    )

    logging.info(f"App loaded. Loading time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
