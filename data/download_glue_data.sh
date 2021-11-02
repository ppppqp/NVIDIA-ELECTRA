#!/bin/bash

data_dir=/workspace/electra/data/download/glue
tasks=all
path_to_mrpc=/workspace/electra/MRPC
python3 download_glue_data.py --data_dir $data_dir --tasks $tasks --path_to_mrpc $path_to_mrpc