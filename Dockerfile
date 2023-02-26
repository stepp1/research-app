# app/Dockerfile
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    graphviz \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*


RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    bash Mambaforge-$(uname)-$(uname -m).sh -b && \
    ~/mambaforge/condabin/conda init && \
    rm Mambaforge-$(uname)-$(uname -m).sh
ENV PATH=$PATH:/root/mambaforge/bin

WORKDIR /app

RUN git clone https://github.com/stepp1/research-app.git .
RUN mamba create -n researcher python=3.10
SHELL ["conda", "run", "-n", "researcher", "/bin/bash", "-c"]

RUN mamba install -n researcher pdfminer matplotlib scikit-learn pandas nltk plotly numpy transformers sentence-transformers fuzzywuzzy umap-learn python-levenshtein pdf2image arxiv pytorch torchvision torchaudio pytorch-cuda=11.7 faiss -c pytorch-nightly -c nvidia -y 

# activate
RUN echo "conda activate researcher" >> ~/.bashrc
RUN echo "Make sure torch is installed:"
RUN python -c "import torch"

RUN python -m pip install streamlit google-search-results selenium selectolax selenium-stealth langchain  InstructorEmbedding python-dotenv "black[jupyter]" openai


RUN echo "Make sure InstructorEmbedding is installed:"
RUN python -c "from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, OpenAIEmbeddings"
RUN echo "Make sure streamlit is installed:"
RUN python -c "import streamlit"

RUN mamba clean --all -y

EXPOSE 8501
EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PATH=/root/mambaforge/envs/researcher/bin:$PATH

CMD ["PYTHONPATH=.", "python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# streamlit run app.py --server.port=8501 --server.address=0.0.0.0
