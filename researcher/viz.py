import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from umap import UMAP

px.defaults.template = "plotly"


def decompose_tsne(embeddings):
    "Creates and TSNE model and plots it"

    tsne_model = TSNE(perplexity=3, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(embeddings)

    return new_values

def decompose_umap(embeddings):
    "Creates and TSNE model and plots it"

    umap_model = UMAP(n_components=2, random_state=42)
    new_values = umap_model.fit_transform(embeddings)

    return new_values


decompose_funcs = {    
    "tsne": decompose_tsne,
    "umap": decompose_umap
}


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
        ax.scatter(x[i],y[i])
        ax.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

    return f

def visualization_plotly(embeddings, labels, cluster_assignment=None, method="tsne"):
    decompose = decompose_funcs[method]
    new_values = decompose(embeddings)

    labels_str = [str(i) for i in labels]
    cluster_assignment_str = [str(i) for i in cluster_assignment]

    fig = px.scatter(
        x = new_values[:, 0], 
        y = new_values[:, 1],
        color = cluster_assignment_str,
        hover_name = labels_str
    )
    # hide x and y axis
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)


    fig.update_layout(showlegend=False)

    fig.update_layout(title="Visualization of Abstract's", height=500, width=1000)
    fig.show()

    return fig
