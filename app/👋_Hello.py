import logging
from functools import partial

import pandas as pd
import streamlit as st

from researcher.parser import arxiv_search

st.set_page_config(page_title="Researcher", page_icon="🧠", layout="wide")
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

st.title("Machine Learning Research: A Paper based Approach")
st.markdown(
    """
    > App by [Stepp1](https://github.com/stepp1)
    >
    > Code available on [Github](https://github.com/stepp1/research-app)
    """
)

st.markdown(
    """
    ### This app is a collection of tools for researchers to explore and understand their papers.
    """,
)

if "source" not in st.session_state:
    st.session_state.source = "None"

if "papers" not in st.session_state:
    st.session_state.papers = pd.DataFrame(
        columns=["Title", "Authors", "Abstract", "URL"]
    )


def center_one_component(component_1):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        out1 = component_1()
    return out1


def center_two_components(component_1, component_2):
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    with col2:
        out1 = component_1()
    with col3:
        out2 = component_2()
    return out1, out2


def set_source():
    # center default papers and custom papers
    out1, out2 = center_two_components(
        partial(st.button, "✅ Default Papers"), partial(st.button, "📁 Custom Papers")
    )
    logging.info(f"out1: {out1}, out2: {out2}")

    if (
        st.session_state.source == "None"
        or st.session_state.source == "default"
        or st.session_state.source == "custom"
    ):
        if out1:
            st.session_state.source = "default"
            st.session_state.papers = pd.read_json(
                "researcher/data/dataset.json", orient="records"
            )

        elif out2:
            st.session_state.source = "custom"
            st.session_state.papers = pd.DataFrame(
                columns=["Title", "Authors", "Abstract", "URL"]
            )

    return None


set_source()
logging.info(f"Source: {st.session_state.source}")

pages = {
    "👋 Hello": "👋_Hello.py",
    "🔎 PaperExplorer": "🔎_PaperExplorer.py",
    "📚 LibraryChat": "📚_LibraryChat.py",
    # "🦜 PaperChat": "🦜_PaperChat.py",
    # "📚 PaperLibrary": "📚_PaperLibrary.py",
    # "📊 PaperMetrics": "📊_PaperMetrics.py",
    # "📈 PaperTrends": "📈_PaperTrends.py",
    # "📝 PaperWriter": "📝_PaperWriter.py",
}

apps = list(pages.keys())[1:]
apps_list = "<br /><br />".join([app for app in apps])


if st.session_state.source == "default":
    st.write(
        '<h4 style="text-align: center;">Hello World of Researcher!</h4>',
        unsafe_allow_html=True,
    )
    st.session_state.papers = pd.read_json("researcher/data/dataset.json")
    st.dataframe(st.session_state.papers, height=200, width=1000)

    st.markdown(
        f'<h4 style="text-align: center;">Just navigate to any of: \n</h4>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h5 style="text-align: center;">{apps_list}!</h5>',
        unsafe_allow_html=True,
    )
elif st.session_state.source == "custom":
    st.write(
        '<h4 style="text-align: center;">Custom Papers:</h4>', unsafe_allow_html=True
    )
    # input Arxiv URL or JSON file
    # make a list of inputs

    input_arxiv = partial(
        st.text_input,
        "🔗 Arxiv URL",
        key="input_arxiv",
        placeholder="https://arxiv.org/abs/11111.00000",
    )

    df_component = partial(
        st.dataframe, st.session_state.papers, height=500, width=1000
    )
    center_one_component(df_component)

    # this is the function the sends the information to that dataframe when called
    # variance is calculated at this point
    def add_paper():
        logging.info(f"Adding paper: {st.session_state.input_arxiv}")

        is_arxiv = "arxiv" in st.session_state.input_arxiv

        if st.session_state.input_arxiv == "":
            return

        elif not is_arxiv:
            return st.warning("Please enter a valid Arxiv URL")

        url = st.session_state.input_arxiv.split("/")[-1]
        # remove the version number
        url = url.split("v")[0]
        paper = arxiv_search(url)

        row = pd.DataFrame(
            {
                "Title": [str(paper["title"])],
                "Authors": [[str(author) for author in paper["authors"]]],
                "Abstract": [paper["abstract"]],
                "URL": [str(paper["url"])],
            }
        )
        st.session_state.papers = pd.concat(
            (st.session_state.papers, row), join="inner", ignore_index=True
        ).drop_duplicates(subset="URL")

    form = st.form(clear_on_submit=True, key="form")

    with form:
        columns = st.columns(1)
        with columns[0]:
            input_arxiv()
        logging.info(f"input_arxiv: {st.session_state.input_arxiv}")

        btn_add = partial(st.form_submit_button, "Submit ➕", on_click=add_paper)
        center_one_component(btn_add)

else:
    pass
