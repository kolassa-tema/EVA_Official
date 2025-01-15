cd preprocess/smplify-x-ehuman-in-the-wild
export PYTHONPATH=$PYTHONPATH:$(pwd)/smplifyx
export PYTHONPATH=$PYTHONPATH:$(pwd)
python script2.py --path ${ROOT_PATH} --out_path ${OUT_PATH} --gpu_id 1 --split_num 40