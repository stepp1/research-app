import streamlit as st

from app.pages.sidebars.library import load_sidebar
from researcher.chat import load_vectorstore, start_conversation

st.header("📚 LibraryChat")

with st.sidebar:
    load_sidebar()


@st.cache_data
def load_response(prompt, source, **kwparams):
    store = load_store(source)
    response = start_conversation(
        prompt,
        chat_history=st.session_state.chat_history,
        vectorstore=store,
        **kwparams
    )
    return response


@st.cache_resource
def load_store(source):
    store = load_vectorstore(source)
    return store


st.write("Welcome to the LibraryChat LLM. Ask a question about a paper in the library!")
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
    hparams = {
        "model": st.session_state.library_model_name,
        "temp": st.session_state.library_model_temp,
    }
    response = load_response(prompt, st.session_state.library_source, **hparams)
    st.session_state.chat_history = (
        st.session_state.chat_history + response["chat_history"]
    )
    st.write(response["answer"])
    print(response)
