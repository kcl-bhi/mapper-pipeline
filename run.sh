#!/bin/bash
# requires: parallel

git clone https://github.com/kcl-bhi/mapper-pipeline.git
python3 -m venv env
source /env
pip install -r requirments.txt
python3 prepare_inputs.py
parallel --progress -j4 < jobs
python3 process_outputs.py

