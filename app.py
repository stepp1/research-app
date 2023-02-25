import logging
import random
import time

import pandas as pd
import streamlit as st

from app.paper_explorer import paper_explorer
from app.sidebar import (
    embeddings_sidebar,
    model_mapping,
    preprocess_sidebar,
    upload_sidebar,
    visualization_sidebar,
)
from researcher.chat import load_vectorstore, start_conversation
from researcher.parser.utils import get_authors_str

st.set_page_config(layout="wide")
random.seed(42)
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

start_time = time.time()
st.title("Machine Learning Research: A Paper based Approach")

current_out_file = "researcher/data/dataset.json"


logging.info(f"Loaded {current_out_file}")


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

    tab1, tab2, tab3 = st.tabs(["📈 Visualization", "🦜PaperChat", "📄 Papers"])

    with tab1:
        data = paper_explorer(data, model, model_mapping, model_name)

    with tab2:
        st.subheader("PaperChat")

        data_source = st.selectbox(
            "Data Source",
            ["Abstract", "Full Text"],
            help="Choose the data source for the pipeline",
            format_func=lambda x: x.title(),
        )

        if data_source == "Abstract":
            source = "abstract"
        else:
            source = "paper"

        llm_model = st.selectbox(
            "Language Model",
            # TODO: keep working on llm.ipynb ["Flan-T5", "OpenAI GPT", "GPT-Neo", "GPT-J"],
            ["OpenAI GPT"],
            help="Choose the language model for the pipeline",
        )

        llm_mapping = {
            "OpenAI GPT": "openai-gpt",
            "GPT-Neo": "EleutherAI/gpt-neo-125M",
            "GPT-J": "EleutherAI/gpt-j-6B",
            "Flan-T5": "google/flan-t5-xxl",
        }
        model_name = llm_mapping[llm_model]
        model_temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Temperature for the language model",
        )

        st.write("Welcome to the PaperChat LLM. Ask a question about a paper!")
        prompt = st.text_input(
            "Prompt/Query:",
            value="List the main takeaways from these papers. Group them by topic.",
        )

        if "count" not in st.session_state or st.session_state.count == 6:
            st.session_state.count = 0
            st.session_state.docs = None
            st.session_state.chat_history = []
        else:
            st.session_state.count += 1
        if prompt != "":
            store = load_vectorstore(source)
            out_dict = start_conversation(
                prompt,
                chat_history=st.session_state.chat_history,
                vectorstore=store,
                model=model_name,
                temp=model_temp,
            )
            st.session_state.chat_history = (
                st.session_state.chat_history + out_dict["chat_history"]
            )
            st.write(out_dict["answer"])
            print(out_dict)

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
