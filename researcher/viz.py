import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

px.defaults.template = "plotly"


def decompose_tsne(embeddings):
    "Creates and TSNE model and plots it"

    tsne_model = TSNE(
        perplexity=3, n_components=2, init="pca", n_iter=2500, random_state=23
    )
    new_values = tsne_model.fit_transform(embeddings)

    return new_values


def decompose_umap(embeddings):
    "Creates and TSNE model and plots it"

    umap_model = UMAP(n_components=2, random_state=42)
    new_values = umap_model.fit_transform(embeddings)

    return new_values


def decompose_pca(embeddings):
    "Creates and TSNE model and plots it"

    pca_model = PCA(n_components=2, random_state=42)
    new_values = pca_model.fit_transform(embeddings)

    return new_values


decompose_funcs = {"tsne": decompose_tsne, "umap": decompose_umap, "pca": decompose_pca}


def visualization_plt(embeddings, labels, method="tsne"):
    decompose = decompose_funcs[method]
    new_values = decompose(embeddings)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    f, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(x)):
        ax.scatter(x[i], y[i])
        ax.annotate(
            labels[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    plt.show()

    return f


def visualization_plotly(
    data=None,
    embeddings=None,
    labels=None,
    color=None,
    decompose_method="tsne",
    show_legend=True,
    show_=False,
    title_add="",
):
    if data is not None:
        embeddings = np.array(data["embeddings"].values.tolist())
        labels = data["label"]
        color = data["color"]
    else:
        assert embeddings is not None
        assert labels is not None
        assert color is not None
        data = pd.DataFrame(
            columns=["component_1", "component_2"],
        )
        data["labels"] = labels
        data["color"] = color

    embeddings = MinMaxScaler().fit_transform(embeddings)
    decompose = decompose_funcs[decompose_method]
    new_values = decompose(embeddings)
    data["component_1"] = new_values[:, 0]
    data["component_2"] = new_values[:, 1]

    labels_str = [str(i) for i in labels]

    fig = px.scatter(
        data_frame=data,
        x="component_1",
        y="component_2",
        color="top_words",
        hover_name=labels_str,
    )
    # hide x and y axis
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)

    fig.update_layout(showlegend=show_legend)

    fig.update_layout(
        title=f"Visualization of Abstract's{title_add}", height=500, width=1000
    )
    if show_:
        fig.show()

    return fig
