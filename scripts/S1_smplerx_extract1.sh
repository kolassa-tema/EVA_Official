#!/bin/bash
cd preprocess/SMPLer-X/main
python script_smplerx2.py --path ${ROOT_PATH} --split_num 2 --gpu_id 2
