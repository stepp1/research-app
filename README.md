# Researcher App
An app that aims to organize your research: *A Researcher with a Paper Based Approach*

Author: @stepp1

[Live Preview]([https://stepp1-research-app-researcherapp-p7i4d2.streamlit.app/](https://huggingface.co/spaces/stepp1/researcher-app))

# Table of Contents
- [Status](#status)
- [Installation](#installation)

# Status
1. What doesn't work
  - Sidebar functionalities
  - Title and Full Text embeddings
  - Better Viz

# Installation
1. Clone the repository
```bash
git clone ...
``` 

2. Install the dependencies using conda/mamba
```bash
conda env create -f environment.yml
```

3. Activate the environment
```bash
conda activate researcher-app
```

4. Run the app
```bash
streamlit run researcher/app.py
```

Remember to forward the port for streamlit if you are running it on a server!

# Data

* PDFs should be stored at researcher/data
* Metadata should be stored at researcher/out/result.json
