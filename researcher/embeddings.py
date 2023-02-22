import logging
import random

import numpy as np
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer

from researcher.models import embeddings_encode

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

    def kmeans_assignment(self, embeddings=None, n_clusters=2):
        self.cluster_method = "kmeans"
        embeddings = self.check_embeddings(embeddings)
        cluster_assignment = embeddings_kmeans(embeddings, n_clusters)
        return cluster_assignment

    def dbscan_assignment(self, embeddings=None, eps=0.5, min_samples=4):
        raise NotImplementedError("DBSCAN not implemented yet")

    def agglomerative_assignment(
        self, embeddings=None, clusters=2, affinity="euclidean", linkage="ward"
    ):
        raise NotImplementedError("Agglomerative Clustering not implemented yet")

    def get_top_words(self, sentences=None, n=4, data=None):
        self.check_sentences(sentences)
        ###### TODO: (step) clean up this code

        # group sentences by cluster
        cluster_sentences = {}
        for i, cluster_idx in enumerate(self.cluster_assignment):
            if cluster not in cluster_sentences:
                cluster_sentences[cluster_idx] = []
            cluster_sentences[cluster_idx].append(sentences[i])

        cluster_tops = {}
        for cluster_idx, cluster_values in cluster_sentences.items():
            # compute tf-idf
            vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_features=1000)
            tfidf = vectorizer.fit_transform(cluster_values)

            # obtain the top 10 words
            top_words = np.array(tfidf.sum(axis=0).tolist()[0]).argsort()[-n:][::-1]

            top_words = [vectorizer.get_feature_names_out()[i] for i in top_words]

            cluster_tops[cluster_idx] = top_words

        top_words = []
        for i, cluster_idx in enumerate(self.cluster_assignment):
            top_words.append(str(cluster_tops[cluster_idx]))

        if data is not None:
            data["top_words"] = top_words
            return data

        return top_words

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


def embeddings_kmeans(embeddings, n_clusters=2):
    clustering_model = cluster.KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10, verbose=0
    )
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment
