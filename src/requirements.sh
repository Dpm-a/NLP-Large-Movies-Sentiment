#!/bin/bash

# Install dependencies
pip install contractions
pip install tqdm
pip install nltk
pip install tensorflow
pip install torch
pip install transformers

# Download nltk model
python -m nltk.downloader stopwords