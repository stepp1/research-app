import logging

import streamlit as st

from researcher.viz import scatterplot


def paper_explorer(data, model, model_mapping, model_name):
    st.subheader("Paper Explorer")

    logging.info("Clustering embeddings")

    # get top words
    data["Top Words per Cluster"] = model.get_top_words(
        sentences=data["sentences_processed"], data=data
    )

    mod_model_name = {v: k for k, v in model_mapping.items()}[model_name]
    title_add = f" - {mod_model_name}"
    fig = scatterplot(
        data_frame=data,
        decompose_method=data.decompose_method,
        color="Top Words per Cluster",
        hover_name="title",
        title_add=title_add,
        return_figure=True,
    )

    logging.info("Plotting visualization")
    st.plotly_chart(fig, use_container_width=True)

    return data
