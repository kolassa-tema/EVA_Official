# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import pickle

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') and
                          not img_fn.startswith('.')]
        self.cnt = 0
        self.dwpose_path = os.path.join(data_folder, 'dwpose.pkl')
        with open(self.dwpose_path, 'rb') as f:
            self.dwpose = pickle.load(f)
        hamer_path = os.path.join(data_folder, 'hamer', 'hamer.pkl')
        with open(hamer_path, 'rb') as f:
            self.hamer = pickle.load(f)

        temp = []
        for x in self.img_paths:
            if x.split('/')[-1] not in self.hamer:
                continue
            if self.hamer[x.split('/')[-1]][0]['pred_keypoints_2d'].shape[0] != 2:
                continue
            smplx_param_path = os.path.join(data_folder, 'smplerx/smplx',
                                            '{:06d}.pkl'.format(int(x.split('/')[-1][:-4])))
            if not os.path.exists(smplx_param_path):
                continue
            temp.append(x)
        self.img_paths = sorted(temp)

        self.avg_shape = np.load(os.path.join(data_folder, 'mean_shape_smplx.npy'))
        self.data_folder = data_folder

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        # optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
        #                         self.use_face * 51 +
        #                         17 * self.use_face_contour,
        #                         dtype=np.float32)
        optim_weights = np.ones(133, dtype=np.float32)
        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        index_key = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1])
        keypoints_dict = self.dwpose[index_key]
        keypoints = np.concatenate((keypoints_dict[0], keypoints_dict[1][:, :, None]), axis=-1)

        # [out, box_center, box_size, batch['right']]
        kp2d_all = self.hamer[img_path.split('/')[-1]][0]['pred_keypoints_2d']
        box_center_all = self.hamer[img_path.split('/')[-1]][1]
        box_size_all = self.hamer[img_path.split('/')[-1]][2]
        is_right_all = self.hamer[img_path.split('/')[-1]][3]
        for i in range(2):
            kp2d = kp2d_all[i]
            cx, cy = box_center_all[i]
            box_size = box_size_all[i]
            is_right = is_right_all[i]
            # unnormalize to crop coords
            kp2d[:, 0] = kp2d[:, 0] * (2 * is_right - 1)
            kp2d = box_size * (kp2d)
            kp2d[:, 0] += cx
            kp2d[:, 1] += cy
            if is_right == 0:
                keypoints[:, 91:112, :2] = kp2d.cpu().numpy()
                keypoints[:, 91:112, 2] = 1
            else:
                keypoints[:, 112:, :2] = kp2d.cpu().numpy()
                keypoints[:, 112:, 2] = 1

        cur_hand = self.hamer[img_path.split('/')[-1]][0]
        cur_left_hand = np.concatenate(
            [cv2.Rodrigues(cur_hand['pred_mano_params']['hand_pose'][0].cpu().numpy()[i])[0] for i in
             range(15)]).squeeze()
        cur_right_hand = np.concatenate(
            [cv2.Rodrigues(cur_hand['pred_mano_params']['hand_pose'][1].cpu().numpy()[i])[0] for i in
             range(15)]).squeeze()
        cur_left_hand = cur_left_hand.reshape(15, 3)[:]
        cur_left_hand[:, 1::3] *= -1
        cur_left_hand[:, 2::3] *= -1
        cur_left_hand = cur_left_hand.reshape(-1)

        cur_hand_3d = cur_hand['pred_keypoints_3d']
        cur_hand_3d = cur_hand_3d.cpu().numpy()
        cam_t = self.hamer[img_path.split('/')[-1]][4]
        cur_hand_3d[0:1, :, 0] = cur_hand_3d[0:1, :, 0] * (-1.)
        cur_hand_3d[0:1] = cur_hand_3d[0:1] + cam_t[0:1, None]
        cur_hand_3d[1:2] = cur_hand_3d[1:2] + cam_t[1:2, None]
        cur_hand_3d = torch.from_numpy(cur_hand_3d.reshape(-1, 3))

        smplx_param_path = os.path.join(self.data_folder, 'smplerx/smplx', '{:06d}.pkl'.format(int(img_path.split('/')[-1][:-4])))
        with open(smplx_param_path, 'rb') as f:
            smplx_param = pickle.load(f)

        smplx_param['left_hand_pose'] = cur_left_hand
        smplx_param['right_hand_pose'] = cur_right_hand
        smplx_param['betas'] = self.avg_shape

        cur_cam_param = np.zeros((3,3))
        cur_cam_param[0][0] = smplx_param['focal'][0]
        cur_cam_param[1][1] = smplx_param['focal'][1]
        cur_cam_param[0][2] = smplx_param['princpt'][0]
        cur_cam_param[1][2] = smplx_param['princpt'][1]
        cur_cam_param[2][2] = 1.0


        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'cam_param': cur_cam_param,
                       'smplx_param': smplx_param,
                       'pGT_lhand': cur_left_hand,
                       'pGT_rhand': cur_right_hand,
                       'p3DGT_hand': cur_hand_3d,
                       'keypoints': keypoints, 'img': img}
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)
