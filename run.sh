#!/bin/bash

git clone https://github.com/Jackson2706/MIND_v2.git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python main.py --data_dir ./data --data_size small --pkl_dir ./pkl --glove_url https://nlp.stanford.edu/data/glove.840B.300d.zip