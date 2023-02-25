import streamlit as st


def load_sidebar():
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

    st.session_state.library_source = source
    st.session_state.library_model_name = model_name
    st.session_state.library_model_temp = model_temp

    return
