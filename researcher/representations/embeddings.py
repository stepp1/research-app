import logging
import random
from collections import Counter

import numpy as np
import pandas as pd

from researcher.representations.cluster import vector_kmeans, vectors_agglomerative
from researcher.representations.models import embeddings_encode

random.seed(42)


class Embeddings:
    def __init__(self, model_name="instructor", key=""):
        self.model_name = model_name
        self.key = key

        self.cluster_assignment = None

        # Embeddings
        self.has_encoded = False
        self.sentences = None
        self.embeddings = None
        self.model = None

        # Clustering Embeddings
        self.cluster_method = ""
        self.cluster_assignment = None

    def encode(self, sentences, clustering=False):
        embeddings, model = embeddings_encode(
            self.model_name, sentences, clustering, self.key
        )

        self.sentences = sentences
        self.embeddings = embeddings
        self.model = model
        self.has_encoded = True
        return embeddings

    def cluster(self, embeddings=None, method_name="kmeans", **kwargs):
        logging.info(f"Clustering with {str(method_name) + '_assignment'}")

        embeddings = self.check_embeddings(embeddings)

        self.cluster_assignment = getattr(self, method_name + "_assignment")(
            embeddings, **kwargs
        )
        return self.cluster_assignment

    def kmeans_assignment(self, embeddings=None, n_clusters=4):
        self.cluster_method = "kmeans"
        embeddings = self.check_embeddings(embeddings)
        cluster_assignment = vector_kmeans(embeddings, n_clusters)
        return cluster_assignment

    def agglomerative_assignment(
        self, embeddings=None, n_clusters=4, linkage="average"
    ):
        self.cluster_method = "agglomerative"
        embeddings = self.check_embeddings(embeddings)
        cluster_assignment = vectors_agglomerative(embeddings, n_clusters, linkage)
        return cluster_assignment

    def dbscan_assignment(self, embeddings=None, eps=0.5, min_samples=4):
        raise NotImplementedError("DBSCAN not implemented yet")

    def get_top_words(self, sentences=None, n=4, data=None):
        self.check_sentences(sentences)

        # build a df in case only sentences is passed
        return_values = False
        if data is None:
            return_values = True
            data = pd.DataFrame(
                {"sentences": sentences, "cluster_assignment": self.cluster_assignment}
            )
        topwords = cluster_topwords(
            data, "cluster_assignment", "sentences_processed", top_k=n, copy=True
        )

        if return_values:
            return topwords.values
        return topwords

    def check_sentences(self, sentences):
        if sentences is None:
            if not self.has_encoded:
                raise ValueError("No sentences provided")
            sentences = self.sentences

        return sentences

    def check_embeddings(self, embeddings):
        if embeddings is None:
            if not self.has_encoded:
                raise ValueError("No embeddings provided")
            embeddings = self.embeddings

        return np.array(embeddings)


def cluster_topwords(df, label_col, sentence_col, top_k=6, copy=True):
    import copy

    df = copy.deepcopy(df)

    df["Top Words per Cluster"] = df[label_col]
    for label in df[label_col].unique():
        cluster_df = df[df[label_col] == label]
        topwords = Counter(" ".join(cluster_df[sentence_col]).split()).most_common(
            top_k
        )

        df.loc[
            df["Top Words per Cluster"] == label, "Top Words per Cluster"
        ] = ", ".join([word[0] for word in topwords])
    return df["Top Words per Cluster"]
