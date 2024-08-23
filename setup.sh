#!/bin/bash

# # Install Python 3.9.0
pyenv local 3.9.0

# Create a virtual environment with Python 3.9.0
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# pip freeze | grep -E "datasets==|transformers==" >> requirements.txt


export OPENAI_API_KEY=your_api_key
# python create_fine_tune_model.py prompt_test7
# python create_dataset.py prompt_test9
# python create_fine_tune_model.py prompt_test7
# python evaluate_fine_tune_model.py prompt_test7 a  > v7_a.txt
