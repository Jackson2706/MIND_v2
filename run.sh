#!/bin/bash

git clone https://github.com/Jackson2706/MIND_v2.git
python main.py --data_dir ./data --data_size small --pkl_dir ./pkl --glove_url https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip