#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import copy
import json
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
import imageio
import cv2
import pickle
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

from smplx.body_models import SMPLX, SMPLX_Rot_mat


INDEX = [233,
 251,
 54,
 72,
 183,
 167,
 253,
 65,
 261,
 22,
 21,
 284,
 203,
 214,
 109,
 265,
 236,
 108,
 73,
 255,
 81,
 270,
 176,
 172,
 223,
 107,
 226,
 259,
 87,
 289,
 15,
 254,
 241,
 83,
 117,
 128,
 30,
 60,
 180,
 50,
 194,
 288,
 256,
 1,
 202,
 76,
 67,
 131,
 43,
 221,
 51,
 116,
 218,
 263,
 160,
 157,
 56,
 20,
 103,
 188,
 27,
 61,
 71,
 179,
 227,
 175,
 268,
 35,
 208,
 166,
 88,
 238,
 113,
 79,
 64,
 178,
 111,
 39,
 286,
 92,
 171,
 28,
 98,
 16,
 136,
 142,
 19,
 229,
 269,
 77,
 6,
 242,
 4,
 0,
 133,
 46,
 138,
 213,
 112,
 120,
 143,
 5,
 279,
 66,
 169,
 266,
 181,
 283,
 55,
 258,
 89,
 84,
 119,
 26,
 211,
 228,
 10,
 68,
 150,
 220,
 141,
 156,
 164,
 215,
 70,
 224,
 7,
 273,
 42,
 271,
 40,
 206,
 86,
 190,
 244,
 274,
 240,
 123,
 186,
 82,
 41,
 148,
 147,
 257,
 144,
 12,
 106,
 158,
 45,
 95,
 104,
 184,
 222,
 246,
 33,
 245,
 277,
 174,
 96,
 90,
 125,
 78,
 216,
 149,
 23,
 122,
 58,
 9,
 132,
 182,
 239,
 260,
 205,
 281,
 146,
 209,
 94,
 29,
 161,
 126,
 31,
 110,
 44,
 139,
 196,
 137,
 124,
 134,
 151,
 130,
 114,
 101,
 2,
 168,
 282,
 53,
 247,
 177,
 47,
 285,
 155,
 219,
 115,
 153,
 252,
 243,
 100,
 34,
 25,
 154,
 48,
 159,
 18,
 102,
 99,
 91,
 267,
 162,
 191,
 93,
 105,
 198,
 3,
 118,
 69,
 278,
 192,
 197,
 37,
 59,
 264,
 173,
 152,
 80,
 200,
 262,
 207,
 250,
 75,
 49,
 248,
 232,
 145,
 97,
 217,
 199,
 276,
 74,
 272,
 275,
 127,
 121,
 85,
 8,
 11,
 187,
 195,
 201,
 13,
 235,
 212,
 129,
 237,
 140,
 193,
 189,
 17,
 230,
 14,
 210,
 280,
 52,
 225,
 38,
 204,
 165,
 135,
 170,
 24,
 287,
 62,
 163,
 57,
 249,
 234,
 36,
 32,
 231,
 63,
 185]

from smplx.utils import Tensor
def batch_rot2aa(
    Rs: Tensor, epsilon: float = 1e-7
) -> Tensor:
    """
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);

        out_theta = std::acos(c);

        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """

    cos = 0.5 * (torch.einsum('bii->b', [Rs]) - 1)
    cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10 + epsilon)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)

class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array
    face_mask: np.array
    lhand_mask: np.array
    rhand_mask: np.array
    face_render_mask: np.array
    lhand_render_mask: np.array
    rhand_render_mask: np.array
    depth: np.array
    kp2d: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

##################################   Real-world   ##################################

def get_camera_extrinsics_ubody_refine(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0, 0, 0, 1]
        ])
        return _viewMatrix

    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi / 2 + np.pi / 12 + 1e-6).astype(
            np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
    return extrinsics

def readCamerasubody(path, output_view, white_background, image_scaling=0.5, split='train',
                              novel_view_vis=False, debug_data=False):
    cam_infos = []

    pose_start = 0
    if split == 'train':
        pose_start = 0
        pose_interval = 2
        pose_num = 140
        if debug_data:
            pose_num = 10 # debug
    elif split == 'test':
        pose_start = 1
        pose_interval = 2
        pose_num = 140
        if debug_data:
            pose_num = 1 # debug

    with open(os.path.join(path, 'dwpose.pkl'), 'rb') as f:
        kp2d_pose = pickle.load(f)

    cams_dict = {}
    ims_dict = {}
    for view_idx in output_view:
        cams = {
            'R': np.eye(3),
            'T': np.zeros((3, 1)),
        }
        cams_dict[view_idx] = cams

        ims_list = sorted(os.listdir(os.path.join(path, 'images')))
        ims = np.array([
            np.array(os.path.join('images', im))
            for im in ims_list[pose_start:pose_start + pose_num * pose_interval][::pose_interval]
        ])

        ims_dict[view_idx] = ims

    ###
    smplx_zoo = {
        'neutral': SMPLX(model_path='../preprocess/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl', ext='pkl',
                      use_face_contour=True, flat_hand_mean=True, use_pca=False,
                      num_betas=10, num_expression_coeffs=10),
    }
    with open(os.path.join(path, 'gender.txt'), 'r') as f:
        gender = f.readline()
    gender = gender.strip()
    smplx_model = smplx_zoo[gender]

    with open('assets/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
        idxs_data = pickle.load(f)
    left_hand = idxs_data['left_hand']
    right_hand = idxs_data['right_hand']
    # hand_idx = np.concatenate([left_hand, right_hand])
    face_idxs = np.load('assets/SMPL-X__FLAME_vertex_ids.npy')

    smpl_param_path = os.path.join(path, 'mean_shape_smplx.npy')
    template_shape = np.load(smpl_param_path)
    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['global_orient'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['betas'] = np.zeros((1, 10)).astype(np.float32)
    big_pose_smpl_param['body_pose'] = np.zeros((1, 63)).astype(np.float32)
    big_pose_smpl_param['jaw_pose'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['left_hand_pose'] = np.zeros((1, 45)).astype(np.float32)
    big_pose_smpl_param['right_hand_pose'] = np.zeros((1, 45)).astype(np.float32)
    big_pose_smpl_param['leye_pose'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['reye_pose'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['expression'] = np.zeros((1, 10)).astype(np.float32)
    big_pose_smpl_param['transl'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['body_pose'][0, 2] = 45 / 180 * np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 5] = -45 / 180 * np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 20] = -30 / 180 * np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 23] = 30 / 180 * np.array(np.pi)
    big_pose_smpl_param_tensor = {}
    for key in big_pose_smpl_param.keys():
        big_pose_smpl_param_tensor[key] = torch.from_numpy(big_pose_smpl_param[key])
    body_model_output = smplx_model(
        global_orient=big_pose_smpl_param_tensor['global_orient'],
        betas=big_pose_smpl_param_tensor['betas'],
        body_pose=big_pose_smpl_param_tensor['body_pose'],
        jaw_pose=big_pose_smpl_param_tensor['jaw_pose'],
        left_hand_pose=big_pose_smpl_param_tensor['left_hand_pose'],
        right_hand_pose=big_pose_smpl_param_tensor['right_hand_pose'],
        leye_pose=big_pose_smpl_param_tensor['leye_pose'],
        reye_pose=big_pose_smpl_param_tensor['reye_pose'],
        expression=big_pose_smpl_param_tensor['expression'],
        transl=big_pose_smpl_param_tensor['transl'],
        return_full_pose=True,
    )
    big_pose_smpl_param['poses'] = body_model_output.full_pose.detach()
    big_pose_smpl_param['shapes'] = np.concatenate([big_pose_smpl_param['betas'], big_pose_smpl_param['expression']],
                                                   axis=-1)
    big_pose_xyz = np.array(body_model_output.vertices.detach()).reshape(-1, 3).astype(np.float32)
    ###

    # # vis
    # from nosmpl.vis.vis_o3d import vis_mesh_o3d
    # vertices = big_pose_xyz.squeeze()
    # faces = smplx_model.faces.astype(np.int32)
    # vis_mesh_o3d(vertices, faces)
    # exit()

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

    idx = 0
    for pose_index in range(pose_num):
        # if split == 'train':
        #     pose_index = 100
        for view_index in output_view:

            if novel_view_vis:
                view_index_look_at = view_index
                view_index = 0

            # Load image, mask, K, D, R, T
            image_path = os.path.join(path, ims_dict[view_index][pose_index].replace('\\', '/'))
            image_name = ims_dict[view_index][pose_index].split('.')[0]
            image = np.array(imageio.imread(image_path).astype(np.float32) / 255.)

            msk_path = image_path.replace('images', 'mask_new').replace('jpg', 'png')
            msk = cv2.imread(msk_path) / 255.0

            if not novel_view_vis:
                R = np.array(cams_dict[view_index]['R'])
                T = np.array(cams_dict[view_index]['T']) #/ 1000.


            # smplx-vposer-0415
            load_cam_param_path = os.path.join(path, 'smplifyx/results', os.path.basename(image_path).split('.')[0] + '.pkl')
            with open(load_cam_param_path, 'rb') as f:
                load_cam_param = pickle.load(f)
            K = load_cam_param['K']

            image = image * msk

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3:4] = T

            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = image_scaling
            # ratio = 1
            if ratio != 1.:
                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                K[:2] = K[:2] * ratio

            image = Image.fromarray(np.array(image * 255.0, dtype=np.byte), "RGB")

            focalX = K[0, 0]
            focalY = K[1, 1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])

            id = os.path.basename(image_path).split('.')[0]
            smpl_param_path = os.path.join(path, 'smplifyx/results', id + '.pkl')
            with open(smpl_param_path, 'rb') as f:
                smpl_param = pickle.load(f)
            for k, v in smpl_param.items():
                smpl_param[k] = v[0]

            smpl_param = {
                'global_orient': np.expand_dims(smpl_param['global_orient'].astype(np.float32), axis=0),
                'transl': np.expand_dims(smpl_param['transl'].astype(np.float32), axis=0),
                'body_pose': np.expand_dims(smpl_param['body_pose'].astype(np.float32), axis=0),
                'jaw_pose': np.expand_dims(smpl_param['jaw_pose'].astype(np.float32), axis=0),
                'betas': np.expand_dims(smpl_param['betas'].astype(np.float32), axis=0),
                'expression': np.expand_dims(smpl_param['expression'].astype(np.float32), axis=0),
                'leye_pose': np.expand_dims(smpl_param['leye_pose'].astype(np.float32), axis=0),
                'reye_pose': np.expand_dims(smpl_param['reye_pose'].astype(np.float32), axis=0),
                'left_hand_pose': np.expand_dims(smpl_param['left_hand_pose'].astype(np.float32), axis=0),
                'right_hand_pose': np.expand_dims(smpl_param['right_hand_pose'].astype(np.float32), axis=0),
                }
            smpl_param['R'] = np.eye(3).astype(np.float32)
            smpl_param['Th'] = smpl_param['transl'].astype(np.float32)
            smpl_param_tensor = {}
            for key in smpl_param.keys():
                smpl_param_tensor[key] = torch.from_numpy(smpl_param[key])
            body_model_output = smplx_model(
                global_orient=smpl_param_tensor['global_orient'],
                betas=smpl_param_tensor['betas'],
                body_pose=smpl_param_tensor['body_pose'],
                jaw_pose=smpl_param_tensor['jaw_pose'],
                left_hand_pose=smpl_param_tensor['left_hand_pose'],
                right_hand_pose=smpl_param_tensor['right_hand_pose'],
                leye_pose=smpl_param_tensor['leye_pose'],
                reye_pose=smpl_param_tensor['reye_pose'],
                expression=smpl_param_tensor['expression'],
                transl=smpl_param_tensor['transl'],
                return_full_pose=True,
            )
            smpl_param['poses'] = body_model_output.full_pose.detach()
            smpl_param['shapes'] = np.concatenate([smpl_param['betas'], smpl_param['expression']], axis=-1)
            xyz = body_model_output.vertices.squeeze().detach().cpu().numpy()

            # from nosmpl.vis.vis_o3d import vis_mesh_o3d
            # vertices = body_model_output.vertices.squeeze()
            # faces = smplx_model.faces.astype(np.int32)
            # vis_mesh_o3d(vertices.detach().cpu().numpy(), faces)
            # vis_mesh_o3d(xyz, faces)
            # ##

            kp2d = np.concatenate([kp2d_pose[image_name+'.png'][0],kp2d_pose[image_name+'.png'][1][:,:,None]], axis=-1)

            # obtain the original bounds for point sampling
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= 0.05
            max_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz], axis=0)

            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
            bound_mask = Image.fromarray(np.array(bound_mask * 255.0, dtype=np.byte))

            try:
                bkgd_mask = Image.fromarray(np.array(msk * 255.0, dtype=np.byte))
            except:
                bkgd_mask = Image.fromarray(np.array(msk[:,:,0] * 255.0, dtype=np.byte))

            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask,
                                        bound_mask=bound_mask, width=image.size[0], height=image.size[1],
                                        smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound,
                                        big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz,
                                        big_pose_world_bound=big_pose_world_bound,
                                        face_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                        lhand_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)), rhand_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                        face_render_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                        lhand_render_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)), rhand_render_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                        kp2d=kp2d,
                                        depth=np.array(bkgd_mask, dtype=np.byte)
                                        ))

            idx += 1

    return cam_infos

def readubodyInfo(path, white_background, output_path, eval, image_scaling=0.5, debug_data=False):
    # all_views = os.listdir(path)
    # all_views = sorted([int(view.split('_')[1][1:]) for view in all_views if ('gender' not in view and 'dwpose' not in view and 'shape' not in view and 'hamer' not in view)])
    train_view = [0]
    test_view = [0]

    print("Reading Training Transforms")
    train_cam_infos = readCamerasubody(path, train_view, white_background, split='train', image_scaling=image_scaling, debug_data=debug_data)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasubody(path, test_view, white_background, split='test', novel_view_vis=False, image_scaling=image_scaling, debug_data=debug_data)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10475  # 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def prepare_smpl_params(smpl_path, pose_index):
    params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
    params = {}
    params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
    params['poses'] = np.zeros((1,72)).astype(np.float32)
    params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
    params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
    params['R'] = np.eye(3).astype(np.float32)
    params['Th'] = np.array(params_ori['transl'][pose_index:pose_index+1]).astype(np.float32)
    return params

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def project_torch(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = torch.matmul(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = torch.matmul(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    corners_2d[:,0] = np.clip(corners_2d[:,0], 0, W)
    corners_2d[:,1] = np.clip(corners_2d[:,1], 0, H)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_2dkps(bounds, K, pose, H, W):
    corners_2d = project(bounds, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    return corners_2d

def get_2dkps_float(bounds, K, pose, H, W):
    corners_2d = project_torch(bounds, K, pose)
    return corners_2d

def get_mask(path, index, view_index, ims):
    msk_path = os.path.join(path, 'mask_cihp',
                            ims[index][view_index])[:-4] + '.png'
    msk_cihp = imageio.imread(msk_path)
    msk_cihp = (msk_cihp != 0).astype(np.uint8)
    msk = msk_cihp.copy()

    return msk, msk_cihp

sceneLoadTypeCallbacks = {
    "ubody": readubodyInfo,
}