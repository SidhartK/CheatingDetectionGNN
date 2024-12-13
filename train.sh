#!/bin/bash

# Assuming you have downloaded the raw_data.csv -> Execute torch-geo-dataset.py
python make_torchgeo_data.py

# Execute python-experiment.py
python main.py
