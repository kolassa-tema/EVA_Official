cd preprocess/smplify-x-ehuman-in-the-wild
export PYTHONPATH=$PYTHONPATH:$(pwd)/smplifyx
export PYTHONPATH=$PYTHONPATH:$(pwd)
python script4.py --path ${ROOT_PATH} --out_path ${OUT_PATH} --gpu_id 3 --split_num 40