tar xvf ckpt.tar.gz
cp -r CKPT_FOLDER/ckpts preprocess/DWPose/control/annotator
cp -r CKPT_FOLDER/pretrained preprocess/SemanticGuidedHumanMatting
cp -r CKPT_FOLDER/pretrained_models preprocess/SMPLer-X
cp -r CKPT_FOLDER/human_model_files preprocess/SMPLer-X/common/utils
cp -r CKPT_FOLDER/_DATA preprocess/hamer

echo "All ckpts moved to desired locations."
