"""
Example Test:
python test_image.py \
    --images-dir "PATH_TO_IMAGES_DIR" \
    --result-dir "PATH_TO_RESULT_DIR" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

Example Evaluation:
python test_image.py \
    --images-dir "PATH_TO_IMAGES_DIR" \
    --gt-dir "PATH_TO_GT_ALPHA_DIR" \
    --result-dir "PATH_TO_RESULT_DIR" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

"""

import argparse
import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image

from model.model import HumanSegment, HumanMatting
import utils
import inference
from tqdm import tqdm
import time

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--pretrained-weight', type=str, required=True)
parser.add_argument('--input_path', type=str, default='')
args = parser.parse_args()

if not os.path.exists(args.pretrained_weight):
    print('Cannot find the pretrained model: {0}'.format(args.pretrained_weight))
    exit()

# --------------- Main ---------------
# Load Model
model = HumanMatting(backbone='resnet50')
model = nn.DataParallel(model).cuda().eval()
model.load_state_dict(torch.load(args.pretrained_weight))
print("Load checkpoint successfully ...")


def extract_one_folder(images_dir, result_dir):
    # Load Images
    image_list = sorted([*glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True),
                        *glob.glob(os.path.join(images_dir, '**', '*.png'), recursive=True)])

    print(images_dir)
    num_image = len(image_list)
    print("Find ", num_image, " images")

    idx_list = []
    # Process
    for i in range(num_image):
        image_path = image_list[i]
        image_name = image_path[image_path.rfind('/')+1:image_path.rfind('.')]

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        # inference
        pred_alpha, pred_mask = inference.single_inference(model, img)

        # save results
        output_dir = result_dir + image_path[len(images_dir):image_path.rfind('/')]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = output_dir + '/' + image_name + '.png'
        Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)

person_path = args.input_path
images_dir = os.path.join(person_path, 'images')
masks_dir = os.path.join(person_path, 'mask_new')
extract_one_folder(images_dir, masks_dir)
print(images_dir, masks_dir)