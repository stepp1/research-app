from sklearn import cluster


def vector_kmeans(vectors, n_clusters=4):
    clustering_model = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        verbose=0,
        max_iter=300,
        tol=0.0001,
        copy_x=True,
        algorithm="lloyd",
    )
    clustering_model.fit(vectors)
    return clustering_model.predict(vectors)


def vectors_agglomerative(
    vectors,
    n_clusters=4,
    metric: str = None,
    use_connnectivity: bool = False,
    linkage="average",
):
    if use_connnectivity:
        # TODO: needs more testing connectivity = sklearn.neighbors.kneighbors_graph(vectors, 3, include_self=False)
        raise NotImplementedError("Connectivity not implemented yet")

    clustering_model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage, metric=metric
    )
    clustering_model.fit(vectors)
    return clustering_model.labels_.astype(int)
