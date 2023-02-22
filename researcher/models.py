import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
)

EMBED_INSTRUCTION_PAPERS = "Represent these paper abstracts for retrieval: "


def load_key(key_name="", key_val=""):
    if Path(".env").exists():
        load_dotenv()
    elif key_val != "":
        os.environ[key_name] = key_val
    else:
        raise ValueError("No .env nor key provided")

    return os.environ[key_name]


@st.cache_resource
def instructor_encode(sentences, clustering=False):
    if clustering:
        embed_instruction = EMBED_INSTRUCTION_PAPERS.replace(
            "for retrieval: ", "for clustering: "
        )
    else:
        embed_instruction = EMBED_INSTRUCTION_PAPERS

    model = HuggingFaceInstructEmbeddings(embed_instruction=embed_instruction)
    embeddings = model.embed_documents(sentences)
    return embeddings, model


@st.cache_resource
def openai_encode(sentences, key=""):
    key = load_key("OPENAI_API_KEY", key)
    model = OpenAIEmbeddings(openai_api_key=key)
    embeddings = model.embed_documents(sentences)
    # os.environ["OPENAI_API_KEY"] = ""

    return embeddings, model


@st.cache_resource
def huggingface_encode(sentences):
    model = HuggingFaceEmbeddings()
    embeddings = model.embed_documents(sentences)
    return embeddings, model


@st.cache_resource
def embeddings_encode(model_name, sentences, clustering=False, key=""):
    if model_name in ["instructor", "instruct"]:
        embeddings = instructor_encode(sentences, clustering)

    elif model_name in ["openai", "open-ai"]:
        embeddings = openai_encode(sentences, key)

    elif model_name in ["huggingface", "hf"]:
        embeddings = huggingface_encode(sentences)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return embeddings
