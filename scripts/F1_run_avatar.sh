cd EVA_main
FOLDER_NAME=$(basename "$ROOT_PATH")
CUDA_VISIBLE_DEVICES=0 python train.py -s ${ROOT_PATH} --eval --exp_name real_world/$FOLDER_NAME --image_scaling 1.0 \
              --smpl_type smplx --actor_gender neutral --iterations 2000