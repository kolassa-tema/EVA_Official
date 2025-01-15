cd preprocess/SemanticGuidedHumanMatting
CUDA_VISIBLE_DEVICES=0 python extract_mask.py --input_path ${ROOT_PATH} --pretrained-weight ./pretrained/SGHM-ResNet50.pth