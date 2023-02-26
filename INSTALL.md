# Installation 

I've always ran into problems when installing from an environtment.yml. Specially, when running in a Docker container.

Therefore, here's a list of the steps I took to install the environment.

First let's clone the repository:
```bash
git clone git@github.com:stepp1/research-app.git
```

And create a `.env` file that holds our API keys:
```bash
touch .env
# Add the following lines to the file
HUGGINGFACEHUB_API_TOKEN = hf_...
OPENAI_API_KEY = sk-...
```

## On your machine

1. Install Mambaforge

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    bash Mambaforge-$(uname)-$(uname -m).sh -b && \
    ~/mambaforge/condabin/conda init && \
    rm Mambaforge-$(uname)-$(uname -m).sh
```

2. Create and activate an environment
```bash
mamba create -n researcher python=3.10 -y
mamba activate researcher
```

3. Install the dependencies
```bash
mamba install python=3.10 pdfminer matplotlib scikit-learn pandas nltk plotly numpy sentence-transformers fuzzywuzzy umap-learn python-levenshtein pdf2image arxiv  pytorch torchvision torchaudio pytorch-cuda=11.7 faiss -c pytorch-nightly -c nvidia -y 

pip install streamlit google-search-results selenium selectolax selenium-stealth langchain "black[jupyter]" InstructorEmbedding python-dotenv
```

4. (Optional) For Jupyter Notebooks
```bash
mamba install jupyterlab ipykernel nbformat -y
```

5. Running the app
```bash
PYTHONPATH=. streamlit run app/👋_Hello.py
# or
bash run.sh
```

6. Open the app in your browser
```bash
http://localhost:8501
```

## On a Docker Container

I provided a Dockerfile to build and run the app.

1. Build the image
```bash
docker build . -t research-app:latest
```

2. Run the container and add the .env file
```bash
docker run --gpus all -i -v $(pwd)/.env:/app/.env -p 8501:8501 -t research-app:latest
```

3. Open the app in your browser
```bash
http://localhost:8501
```
