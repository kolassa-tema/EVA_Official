import os
import cv2
from dwpose.wholebody import Wholebody
from tqdm import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='')
args = parser.parse_args()

pose_detector = Wholebody()

id_path = args.input_path
results_dict = {}
img_list = sorted(os.listdir(os.path.join(id_path, 'images')))
img_list = sorted([x for x in img_list if 'png' in x])
for img_name in tqdm(img_list):
    img_path = os.path.join(id_path, 'images', img_name)
    img = cv2.imread(img_path)
    result = pose_detector(img)
    results_dict[os.path.join('images', img_name)] = result
with open(os.path.join(id_path, 'dwpose.pkl'), 'wb') as f:
    pickle.dump(results_dict, f)
