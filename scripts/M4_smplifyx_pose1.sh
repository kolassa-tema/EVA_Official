cd preprocess/smplify-x-ehuman-in-the-wild
export PYTHONPATH=$PYTHONPATH:$(pwd)/smplifyx
export PYTHONPATH=$PYTHONPATH:$(pwd)
python script1.py --path ${ROOT_PATH} --out_path ${OUT_PATH} --gpu_id 0 --split_num 40