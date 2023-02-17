import logging
import os
import random
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
)
from sklearn import cluster

random.seed(42)

EMBED_INSTRUCTION_PAPERS = "Represent these paper abstracts for retrieval: "


class Embeddings:
    def __init__(self, model_name="instructor", key=""):
        self.model_name = model_name
        self.key = key

    def encode(self, sentences, clustering=False):
        embeddings, model = embeddings_encode(
            self.model_name, sentences, clustering, self.key
        )
        self.model = model
        return embeddings

    def cluster_assignment(self, embeddings, method_name="kmeans", **kwargs):
        logging.info("Clustering with {}".format(str(method_name) + "_assignment"))
        return getattr(self, method_name + "_assignment")(embeddings, **kwargs)

    def kmeans_assignment(self, embeddings, n_clusters=2):
        return embeddings_kmeans(embeddings, n_clusters)

    def dbscan_assignment(self, embeddings, eps=0.5, min_samples=4):
        raise NotImplementedError("DBSCAN not implemented yet")

    def agglomerative_assignment(
        self, embeddings, clusters=2, affinity="euclidean", linkage="ward"
    ):
        raise NotImplementedError("Agglomerative Clustering not implemented yet")


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
        raise ValueError("Unknown model name: {}".format(model_name))

    return embeddings


def embeddings_kmeans(embeddings, n_clusters=2):
    clustering_model = cluster.KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10, verbose=0
    )
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment
