import json
import random
from parser.parse import extract_paper
from parser.utils import add_to_json, get_authors_str

import streamlit as st
from embeddings import instructor_encode, kmeans_embeddings, load_model
from viz import visualization_plotly

st.set_page_config(layout="wide")
random.seed(42)

st.title("Machine Learning Research: A Paper based Approach")

data_source = st.selectbox(
    "Data Source (Currently only Abstracts are supported)",
    ["Abstract"], # ["Title", "Abstract", "Full Text"],
    help = "Choose the data source for the pipeline",
    format_func=lambda x: x.title()
)
if data_source == "Abstract":
    current_out_file = "researcher/out/result.json"
else:
    current_out_file = "researcher/out/result.json"

with st.sidebar:
    with st.expander("Want to add a new Paper?"):
        uploaded_file = st.file_uploader("Upload a Paper's PDF", type="pdf")
        if uploaded_file is not None:
            st.write("File uploaded!")
            
            # parse PDF and extract paper
            result = extract_paper(uploaded_file.name, "papers.json")

            add_to_json(result, current_out_file)
    
    st.title("Pipeline Configuration")
    
    ## preprocessing
    with st.expander("Preprocessing", False):
        prep = data_source
        pre = st.multiselect(
            "Preprocessing Steps", 
            ["Lowercase","Punctuation","Stopwords","Lemmatization","Stemming","Spelling","Clause Separation"], 
            help = "Preprocessing steps to apply to provided data"
        )
        # prep_load()

        # if "Lowercase" in pre:
        #     prep = prep_lower(prep)
        # if "Punctuation" in pre:
        #     prep = prep_punct(prep)
        # if "Stopwords" in pre:
        #     prep = prep_stop(prep)
        # if "Lemmatization" in pre:
        #     prep = prep_lemma(prep)
        # if "Stemming" in pre:
        #     prep = prep_stem(prep)
        # if "Spelling" in pre:
        #     prep = prep_spell(prep)
        # if "Clause Separation" in pre:
        #     clause_reg_box = st.text_input("clause sep regex", clause_reg, help = "Regex defining separation of clauses within each sentence/line")
        #     clause_word_box = st.text_input("clause sep words", clause_words, help = "Words indicating a clause boundary")
        #     clause_sep = f"{clause_reg}{' | '.join(clause_words)}".replace("] ", "]")
        #     prep = prep_clause(prep)

    ## embeddings
    with st.expander("Embeddings", False):
        mod = st.selectbox(
            "Embedding Algorithm", 
            ["tf-idf","Hash","Count","SentenceTransformers Model","Universal Sentence Encoder"], 
            help = "Algorithm used for sentence embeddings, preprocessing steps may be duplicated between the abova and the following models"
        )

        if mod == "tf-idf":
            ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
            # emb = model_tfidf(prep, (1,ng))
        if mod == "Hash":
            ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
            # emb = model_hash(prep, (1,ng))
        if mod == "Count":
            ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
            # emb = model_count(prep, (1,ng))
        # if mod == "SentenceTransformers Model":
            # st_mod = st.selectbox("st model selection", st_available_models, help = "Pretrained models available through the SetnenceTransformers library and HuggingFace.co")
            # emb = model_snt(prep, st_mod)
        if mod == "Universal Sentence Encoder":
            # emb = model_use(prep)
            ...
    ## viz 
    with st.expander("Visualization", False):
        mod = st.selectbox(
            "Dimensionality Reduction", 
            ["PCA","t-SNE","UMAP"],
            help = "Algorithm used to reduce the dimensionality of the embeddings"
        )

        decompose_method = mod.lower().replace("-","")

        mod = st.selectbox(
            "Clustering Algorithm", 
            ["K-Means","DBSCAN","Agglomerative Clustering"],
            help = "Algorithm used to cluster the embeddings"
        )

        
        if mod == "K-Means":
            cluster_method = "kmeans"
            n_clusters = st.slider("n_clusters", 1, 10, value=4, help = "Number of clusters to use for k-means clustering")
        if mod == "DBSCAN":
            cluster_method = "dbscan"
            eps = st.slider("eps", 0.0, 1.0, value=0.5, help = "The maximum distance between two samples for one to be considered as in the neighborhood of the other")
            min_samples = st.slider("min_samples", 1, 10, value=4, help = "The number of samples (or total weight) in a neighborhood for a point to be considered as a core point")
        if mod == "Agglomerative Clustering":
            cluster_method = "agglomerative"
            n_clusters = st.slider("n_clusters", 1, 10, value=4, help = "The number of clusters to find")
            affinity = st.selectbox("affinity", ["euclidean","l1","l2","manhattan","cosine","precomputed"], help = "Metric used to compute the linkage")
            linkage = st.selectbox("linkage", ["ward","complete","average","single"], help = "Which linkage criterion to use")


            
tab1, tab2 = st.tabs(["📈 Visualization", "🗃 Data"])

with open(current_out_file, "r") as f:
        papers = json.load(f)

with tab1:
    st.subheader("Paper Explorer")

    abstracts = [paper["abstract"] for paper in papers]
    model = load_model()
    embeddings = instructor_encode(model, abstracts, clustering=True)
    cluster_assignment = kmeans_embeddings(embeddings, n_clusters)

    fig = visualization_plotly(
        embeddings=embeddings, 
        labels=[paper["title"] for paper in papers], 
        cluster_assignment=cluster_assignment, 
        method=decompose_method,
        show_legend=True,
        show_=False
    )

    st.plotly_chart(fig)

with tab2:
    st.subheader("Current Papers")

    # 3 columns 
    col1, col2, col3 = st.columns(3)

    for i, paper in enumerate(papers):
        if i % 3 == 0:
            current_col = col1    
        elif i % 3 == 1:
            current_col = col2
        else:
            current_col = col3
        
        current_col.write(f"**{paper['title']}**")
        current_col.write(f"Authors: {get_authors_str(paper['authors'])}")
        current_col.write(f"Abstract: {paper['abstract']}")
        current_col.write(f"Link: {paper['url']}")