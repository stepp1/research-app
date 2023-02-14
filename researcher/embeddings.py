import random

from InstructorEmbedding import INSTRUCTOR
from sklearn import cluster

random.seed(42)


def load_model(model_name = "hkunlp/instructor-large"):
    model = INSTRUCTOR(model_name)
    return model

def instructor_encode(model, sentences, clustering = False):
    instruction = 'Represent these paper abstracts: '

    if clustering:
        instruction = instruction.replace(": ", "for clustering: ")

    sentences = [[instruction, sentence] for sentence in sentences]
    embeddings = model.encode(sentences)
    return embeddings

def kmeans_embeddings(embeddings, clusters = 2):
    clustering_model = cluster.KMeans(
        n_clusters=clusters, 
        random_state=42, 
        n_init=10, 
        verbose=0
    )
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment