#!/bin/bash

git clone https://github.com/Jackson2706/MIND_v2.git
pip install -r requirements.txt
python main.py --data_dir ./data --data_size small --pkl_dir ./pkl --glove_url https://nlp.stanford.edu/data/glove.840B.300d.zip