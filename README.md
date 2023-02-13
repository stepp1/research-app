# Researcher App
An app to organize your research for you: A Researcher with a Paper Based Approach

Author: @stepp1

# Table of Contents
- [Installation](#installation)

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