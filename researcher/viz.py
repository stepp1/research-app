import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

px.defaults.template = "plotly"


def decompose_tsne(vector, return_series=False):
    "Creates and TSNE model and plots it"
    if isinstance(vector, pd.Series):
        return_series = True
        vector = np.vstack(vector.values)

    tsne_model = TSNE(
        n_components=2,
        perplexity=3,
        init="pca",
        min_grad_norm=1e-07,
        metric="euclidean",
        random_state=42,
        n_iter=1000,
        n_iter_without_progress=300,
        method="barnes_hut",
        angle=0.5,
        n_jobs=-1,
    )
    new_values = tsne_model.fit_transform(vector)

    if return_series:
        new_values = pd.Series(new_values)

    return new_values


def decompose_umap(vector, return_series=False):
    "Decomposes the embeddings using UMAP"
    if isinstance(vector, pd.Series):
        return_series = True
        vector = np.vstack(vector.values)

    umap_model = UMAP(n_components=2, random_state=42)
    new_values = umap_model.fit_transform(vector)

    if return_series:
        new_values = pd.Series(new_values)

    return new_values


def decompose_pca(vector, return_series=False):
    "Decomposes the embeddings using PCA"
    if isinstance(vector, pd.Series):
        return_series = True
        vector = np.vstack(vector.values)

    pca_model = PCA(n_components=2, random_state=42)
    new_values = pca_model.fit_transform(vector)

    if return_series:
        new_values = pd.Series(new_values)

    return new_values


decompose_funcs = {"tsne": decompose_tsne, "umap": decompose_umap, "pca": decompose_pca}


def scatterplot(
    data_frame=None,
    embeddings=None,
    hover_name="label",
    labels=None,
    color=None,
    decompose_method="tsne",
    show_legend=True,
    return_figure=False,
    title_add="",
):
    if data_frame is not None:
        embeddings = data_frame["embeddings"].to_numpy()
        if hover_name == "label":
            hover_name = data_frame["label"].to_numpy()

    else:
        try:
            assert embeddings is not None
            assert labels is not None
            assert color is not None
        except AssertionError:
            raise AssertionError(
                "You must pass either a data_frame or embeddings, labels, and color"
            )

        data_frame = pd.DataFrame(
            columns=["component_1", "component_2"],
        )
        data_frame[hover_name] = labels
        data_frame["color"] = color

    embeddings = MinMaxScaler().fit_transform(embeddings.tolist())
    decompose = decompose_funcs[decompose_method]
    new_values = decompose(embeddings)
    data_frame["component_1"] = new_values[:, 0]
    data_frame["component_2"] = new_values[:, 1]

    fig = px.scatter(
        data_frame=data_frame,
        x="component_1",
        y="component_2",
        color=color,
        hover_name=hover_name,
    )
    # hide x and y axis
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)

    # hide legend
    fig.update_layout(showlegend=show_legend)

    # custom title and size
    fig.update_layout(
        title=f"Visualization of Abstract's{title_add}", height=500, width=1000
    )

    if return_figure:
        return fig

    fig.show()
