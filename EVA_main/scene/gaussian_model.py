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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from knn_cuda import KNN
import pickle
import torch.nn.functional as F
from nets.mlp_delta_body_pose import EBodyPoseRefiner
from nets.mlp_delta_weight_lbs import LBSOffsetDecoder
from nets.refine_sr import ResidualDenseBlock_Conf
from smplx.body_models import SMPLX


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, transform):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if transform is not None:
                actual_covariance = transform @ actual_covariance
                actual_covariance = actual_covariance @ transform.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, smpl_type : str, motion_offset_flag : bool, actor_gender: str, epose: bool, shs_corr: bool,):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._view_ind_feats = torch.empty(0) # **************
        self._xyz_gradient_prev = torch.empty(0) # **************
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.view_ind_channels = 1
        self.semantic_add = None
        self.semantic_grad = False
        self.track_first_h_lamuda = True
        self.shs_corr = shs_corr

        self.device=torch.device('cuda', torch.cuda.current_device())
        # load SMPL model
        self.gender = actor_gender
        if smpl_type == 'smpl':
            neutral_smpl_path = os.path.join('assets', f'SMPL_{actor_gender.upper()}.pkl')
            self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=self.device)
        elif smpl_type == 'smplx':
            male_smpl_path = os.path.join('../preprocess/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_{}.pkl'.format(actor_gender.upper()))
            with open(male_smpl_path, 'rb') as f:
                params_init = pickle.load(f, encoding='latin1')
            self.SMPL_NEUTRAL = SMPL_to_tensor(params_init, device=self.device)
            smplx_model = SMPLX(model_path='../preprocess/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_{}.pkl'.format(actor_gender.upper()), ext='pkl',
                                use_face_contour=True, flat_hand_mean=False, use_pca=False,
                                num_betas=10, num_expression_coeffs=10)
            new_shapedirs = torch.cat([smplx_model.shapedirs, smplx_model.expr_dirs], dim=-1)
            self.SMPL_NEUTRAL['shapedirs'] = new_shapedirs.clone().to(device=self.device)

        # load knn module
        self.knn = KNN(k=1, transpose_mode=True)
        self.knn_near_2 = KNN(k=2, transpose_mode=True)

        self.motion_offset_flag = motion_offset_flag
        if self.motion_offset_flag:
            # load pose correction module
            total_bones = self.SMPL_NEUTRAL['weights'].shape[-1]
            self.pose_decoder = EBodyPoseRefiner(total_bones=total_bones, embedding_size=3*(total_bones-1), mlp_width=128, mlp_depth=2)
            self.pose_decoder.to(self.device)

            # load lbs weight module
            self.lweight_offset_decoder = LBSOffsetDecoder(total_bones=total_bones)
            self.lweight_offset_decoder.to(self.device)

            self.refine_net = ResidualDenseBlock_Conf(4)
            self.refine_net.to(self.device)
            self.view_embed_list = nn.Embedding(200, 16)
            self.view_embed_list.to(self.device)
            self.gp_loss_mult = 0.0003


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._view_ind_feats, # ***************
            self._xyz_gradient_prev,  # ***************
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.pose_decoder,
            self.lweight_offset_decoder,
        )
    
    def restore(self, model_args, training_args):
        # check restore
        print('restore')
        (self.active_sh_degree,
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._view_ind_feats, # **************
        self._xyz_gradient_prev, # **************
        self._scaling,
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.pose_decoder,
        self.lweight_offset_decoder) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    # *******************
    @property
    def get_view_ind_feats(self):
        return self._view_ind_feats

    # *******************
    @property
    def get_xyz_gradient_prev(self):
        return self._xyz_gradient_prev

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1, transform=None):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, transform)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, semantic_add=False):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.004 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # ori 0.1, 0.004

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if semantic_add:
            self.semantic_add = semantic_add
            with open('assets/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
                idxs_data = pickle.load(f)
                left_hand = idxs_data['left_hand']
                right_hand = idxs_data['right_hand']
                hand_idx = np.concatenate([left_hand, right_hand])
            face_idxs = np.load('assets/SMPL-X__FLAME_vertex_ids.npy')
            whole = torch.zeros(fused_color.shape[0], self.view_ind_channels)
            whole[hand_idx] = 1
            whole[face_idxs] = 2
            self.template_view_ind_feats = whole.detach().clone()
            self._view_ind_feats = nn.Parameter(whole.float().cuda().contiguous().requires_grad_(True)) # *******here should be N,1,C or N,C?
        else:
            self._view_ind_feats = nn.Parameter(torch.zeros(fused_color.shape[0], self.view_ind_channels).float().cuda().contiguous().requires_grad_(True)) # *******here should be N,1,C or N,C?
            self.template_view_ind_feats = None
        self._xyz_gradient_prev = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 1)).float().cuda().contiguous().requires_grad_(False))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if not self.motion_offset_flag:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self._view_ind_feats], 'lr': training_args.feature_lr, "name": "f_view_ind"},
                {'params': [self._xyz_gradient_prev], 'lr': training_args.feature_lr, "name": "xyz_gradient_prev"},
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self._view_ind_feats], 'lr': training_args.feature_lr, "name": "f_view_ind"},
                {'params': [self._xyz_gradient_prev], 'lr': training_args.feature_lr, "name": "xyz_gradient_prev"},
                {'params': self.pose_decoder.body_refiner.parameters(), 'lr': training_args.pose_refine_lr_init, "name": "pose_decoder_body"},
                {'params': self.pose_decoder.face_refiner.parameters(), 'lr': training_args.pose_refine_lr_init,  "name": "pose_decoder_face"},
                {'params': self.pose_decoder.hand_refiner.parameters(), 'lr': training_args.pose_refine_lr_init, "name": "pose_decoder_hand"},
                {'params': self.lweight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr, "name": "lweight_offset_decoder"},
                {'params': self.refine_net.parameters(), 'lr': 1e-3, "name": "refine_net"},
                {'params': self.view_embed_list.parameters(), 'lr': 5e-5, "name": "view_embed_list"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.pose_refine_scheduler_args = get_expon_lr_func(lr_init=training_args.pose_refine_lr_init,
                                                    lr_final=training_args.pose_refine_lr_final,
                                                    lr_delay_steps=training_args.pose_refine_lr_delay_steps,
                                                    lr_delay_mult=training_args.pose_refine_lr_delay_mult,
                                                    max_steps=training_args.pose_refine_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if 'pose_decoder' in param_group['name']:
                if 'hand' in param_group['name'] or 'face' in param_group['name']:
                    lr = self.pose_refine_scheduler_args(iteration) * 0.05
                else:
                    lr = self.pose_refine_scheduler_args(iteration)
                param_group['lr'] = lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._view_ind_feats.shape[1]):
            l.append('f_view_ind_{}'.format(i))
        for i in range(self._xyz_gradient_prev.shape[1]):
            l.append('xyz_gradient_prev_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_view_ind = self._view_ind_feats.detach().cpu().numpy()
        xyz_gradient_prev = self._xyz_gradient_prev.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, f_view_ind, xyz_gradient_prev, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        f_view_ind_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_view_ind_')]
        f_view_ind_names = sorted(f_view_ind_names, key=lambda x: int(x.split('_')[-1]))
        f_view_ind = np.zeros((xyz.shape[0], len(f_view_ind_names)))
        for idx, attr_name in enumerate(f_view_ind_names):
            f_view_ind[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz_gradient_prev_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('xyz_gradient_prev_')]
        xyz_gradient_prev_names = sorted(xyz_gradient_prev_names, key=lambda x: int(x.split('_')[-1]))
        xyz_gradient_prev = np.zeros((xyz.shape[0], len(xyz_gradient_prev_names)))
        for idx, attr_name in enumerate(xyz_gradient_prev_names):
            xyz_gradient_prev[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._view_ind_feats = nn.Parameter(torch.tensor(f_view_ind, dtype=torch.float, device='cuda').requires_grad_(True))
        self._xyz_gradient_prev = nn.Parameter(torch.tensor(xyz_gradient_prev, dtype=torch.float, device='cuda').requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # if len(group["params"]) == 1:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'f_view_ind', 'xyz_gradient_prev']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._view_ind_feats = optimizable_tensors['f_view_ind'] # ****************
        self._xyz_gradient_prev = optimizable_tensors['xyz_gradient_prev'] # ****************
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group["params"]) == 1
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'f_view_ind', 'xyz_gradient_prev']:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_view_ind_feats, new_xyz_gradient_prev, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "f_view_ind": new_view_ind_feats,  # *****************
            'xyz_gradient_prev': new_xyz_gradient_prev, # *****************
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._view_ind_feats = optimizable_tensors['f_view_ind'] # *************************
        self._xyz_gradient_prev = optimizable_tensors['xyz_gradient_prev'] # *************************

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, body_alpha, hand_alpha, face_alpha, lambda_pos_grad, N=2):

        def densify_and_split_single(grads, grad_threshold, scene_extent, indicator, alpha, lambda_pos_grad, N=2):
            n_init_points = self.get_xyz.shape[0]
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()

            if self.semantic_grad:
                if self.track_first_h_lamuda:
                    grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * (
                                torch.norm(grads, dim=-1).mean() - torch.norm(self.grads_prev, dim=-1).mean())
                elif indicator == 0:
                    grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * \
                                     (padded_grad[torch.where(self.get_view_ind_feats == indicator, True, False).squeeze()].mean() -
                                                                                 self.grads_prev_body.mean())
                elif indicator == 1:
                    grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * \
                                     (padded_grad[torch.where(self.get_view_ind_feats == indicator, True, False).squeeze()].mean() -
                                                                                 self.grads_prev_hand.mean())
                elif indicator == 2:
                    grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * \
                                     (padded_grad[torch.where(self.get_view_ind_feats == indicator, True, False).squeeze()].mean() -
                                                                                self.grads_prev_face.mean())
            else:
                grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * (padded_grad.mean() - self.grads_prev.mean())

            selected_pts_mask = torch.where(padded_grad >= grad_thres_cur, True, False)

            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.where(self.get_view_ind_feats == indicator, True, False)[:,0])
            return selected_pts_mask

        body_selected_pts_mask = densify_and_split_single(grads, grad_threshold, scene_extent, indicator=0, lambda_pos_grad=lambda_pos_grad, alpha=body_alpha)
        hand_selected_pts_mask = densify_and_split_single(grads, grad_threshold, scene_extent, indicator=1, lambda_pos_grad=lambda_pos_grad, alpha=hand_alpha)
        face_selected_pts_mask = densify_and_split_single(grads, grad_threshold, scene_extent, indicator=2, lambda_pos_grad=lambda_pos_grad, alpha=face_alpha)
        selected_pts_mask = body_selected_pts_mask | hand_selected_pts_mask | face_selected_pts_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_view_ind_feats = self._view_ind_feats[selected_pts_mask].repeat(N,1) # *****************************
        new_xyz_gradient_prev = self._xyz_gradient_prev[selected_pts_mask].repeat(N,1) # *****************************

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_view_ind_feats, new_xyz_gradient_prev, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, body_alpha, hand_alpha, face_alpha, lambda_pos_grad):
        # Extract points that satisfy the gradient condition
        def densify_and_clone_single(grads, grad_threshold, scene_extent, indicator, alpha, lambda_pos_grad):
            if self.semantic_grad:
                if self.track_first_h_lamuda:
                    grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * (
                                torch.norm(grads, dim=-1).mean() - torch.norm(self.grads_prev, dim=-1).mean())
                elif indicator == 0:
                    grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * \
                                     (torch.norm(grads[torch.where(self.get_view_ind_feats == indicator, True, False)], dim=-1).mean() -
                                                                                 torch.norm(self.grads_prev_body, dim=-1).mean())
                elif indicator == 1:
                    grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * \
                                     (torch.norm(grads[torch.where(self.get_view_ind_feats == indicator, True, False)], dim=-1).mean() -
                                                                                 torch.norm(self.grads_prev_hand, dim=-1).mean())
                elif indicator == 2:
                    grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * \
                                     (torch.norm(grads[torch.where(self.get_view_ind_feats == indicator, True, False)], dim=-1).mean() -
                                                                                 torch.norm(self.grads_prev_face, dim=-1).mean())
            else:
                grad_thres_cur = alpha * grad_threshold + alpha * lambda_pos_grad * (torch.norm(grads, dim=-1).mean() - torch.norm(self.grads_prev, dim=-1).mean())

            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_thres_cur, True, False)

            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.where(self.get_view_ind_feats == indicator, True, False)[:,0])
            return selected_pts_mask

        # full
        body_selected_pts_mask = densify_and_clone_single(grads, grad_threshold, scene_extent, indicator=0, lambda_pos_grad=lambda_pos_grad, alpha=body_alpha)
        hand_selected_pts_mask = densify_and_clone_single(grads, grad_threshold, scene_extent, indicator=1, lambda_pos_grad=lambda_pos_grad, alpha=hand_alpha)
        face_selected_pts_mask = densify_and_clone_single(grads, grad_threshold, scene_extent, indicator=2, lambda_pos_grad=lambda_pos_grad, alpha=face_alpha)
        selected_pts_mask = body_selected_pts_mask | hand_selected_pts_mask | face_selected_pts_mask

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_view_ind_feats = self._view_ind_feats[selected_pts_mask] # **************
        new_xyz_gradient_prev = self._xyz_gradient_prev[selected_pts_mask] # **************

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_view_ind_feats, new_xyz_gradient_prev, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,
                          body_alpha=1.0, hand_alpha=1.0, face_alpha=1.0,
                          kl_threshold=None, t_vertices=None, iter=None,
                          lambda_split_pos_grad=0, lambda_clone_pos_grad=0, ema=0, semantic_grad=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # ==========================================================================================================================  1220_exp1_(a): Heuristic
        if self.track_first_h_lamuda == True:
            self.grads_prev = torch.zeros_like(grads)
            self.grads_prev_body, self.grads_prev_hand, self.grads_prev_face = 0., 0., 0.
        # ==========================================================================================================================
        self.cur_gs_sem = self.get_view_ind_feats.detach().clone()
        self.semantic_grad = semantic_grad

        self.densify_and_clone(grads, max_grad, extent, body_alpha=body_alpha, hand_alpha=hand_alpha, face_alpha=face_alpha, lambda_pos_grad=lambda_clone_pos_grad)
        self.densify_and_split(grads, max_grad, extent, body_alpha=body_alpha, hand_alpha=hand_alpha, face_alpha=face_alpha, lambda_pos_grad=lambda_split_pos_grad)

        self.track_first_h_lamuda = False
        self.grads_prev = grads.detach().clone()  # Heuristic
        self.grads_prev_body = self.grads_prev[torch.where(self.cur_gs_sem == 0, True, False)]
        self.grads_prev_hand = self.grads_prev[torch.where(self.cur_gs_sem == 1, True, False)]
        self.grads_prev_face = self.grads_prev[torch.where(self.cur_gs_sem == 2, True, False)]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)


        if self.template_view_ind_feats is None:
            # use smpl prior to prune points
            distance, _ = self.knn(t_vertices[None], self._xyz[None].detach())
            distance = distance.view(distance.shape[0], -1)
            threshold = 0.05
            pts_mask = (distance > threshold).squeeze()
            all_prune_mask = prune_mask | pts_mask
        else:
            distance, _ = self.knn(t_vertices[None], self._xyz[None].detach())
            distance = distance.view(distance.shape[0], -1)
            threshold = 0.05
            pts_mask = (distance > threshold).squeeze()

            t_vertices_body = t_vertices[torch.where(self.template_view_ind_feats.squeeze() == 0, True, False)]
            cur_vertices_body = self._xyz[torch.where(self.get_view_ind_feats.squeeze() == 0, True, False)]
            distance_body, _ = self.knn(t_vertices_body[None], cur_vertices_body[None].detach())
            distance_body = distance_body.view(distance_body.shape[0], -1)
            threshold_body = 0.03
            pts_mask_body = (distance_body > threshold_body).squeeze()
            pts_mask[torch.nonzero(torch.where(self.get_view_ind_feats.squeeze() == 0, True, False).squeeze()==True).squeeze()] = pts_mask_body

            t_vertices_hand = t_vertices[torch.where(self.template_view_ind_feats.squeeze() == 1, True, False)]
            cur_vertices_hand = self._xyz[torch.where(self.get_view_ind_feats.squeeze() == 1, True, False)]
            distance_hand, _ = self.knn(t_vertices_hand[None], cur_vertices_hand[None].detach())
            distance_hand = distance_hand.view(distance_hand.shape[0], -1)
            threshold_hand = 0.05
            pts_mask_hand = (distance_hand > threshold_hand).squeeze()
            pts_mask[torch.nonzero(torch.where(self.get_view_ind_feats.squeeze() == 1, True, False).squeeze()==True).squeeze()] = pts_mask_hand

            t_vertices_face = t_vertices[torch.where(self.template_view_ind_feats.squeeze() == 2, True, False)]
            cur_vertices_face = self._xyz[torch.where(self.get_view_ind_feats.squeeze() == 2, True, False)]
            distance_face, _ = self.knn(t_vertices_face[None], cur_vertices_face[None].detach())
            distance_face = distance_face.view(distance_face.shape[0], -1)
            threshold_face = 0.09
            pts_mask_face = (distance_face > threshold_face).squeeze()
            pts_mask[torch.nonzero(torch.where(self.get_view_ind_feats.squeeze() == 2, True, False).squeeze() == True).squeeze()] = pts_mask_face

            all_prune_mask = prune_mask | pts_mask


        print('total points num: ', self._xyz.shape[0], 'prune num: ', all_prune_mask.sum().item(),
              prune_mask.sum().item(), pts_mask.sum().item())
        
        self.prune_points(all_prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def coarse_deform_c2source(self, query_pts, params, t_params, t_vertices, lbs_weights=None, correct_Rs=None, return_transl=False):
        bs = query_pts.shape[0]
        joints_num = self.SMPL_NEUTRAL['weights'].shape[-1]
        vertices_num = t_vertices.shape[1]
        # Find nearest smpl vertex
        smpl_pts = t_vertices

        _, vert_ids = self.knn(smpl_pts.float(), query_pts.float())
        if lbs_weights is None:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)#.cuda() # [bs, points_num, joints_num]
        else:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)
            bweights = torch.log(bweights + 1e-9) + lbs_weights
            bweights = F.softmax(bweights, dim=-1)

        ### From Big To T Pose
        big_pose_params = t_params
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        query_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        query_pts = torch.matmul(R_inv, query_pts[..., None]).squeeze(-1)

        # transforms from Big To T Pose
        transforms = R_inv

        # translation from Big To T Pose
        translation = None
        if return_transl: 
            translation = -A[..., :3, 3]
            translation = torch.matmul(R_inv, translation[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = big_pose_params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts - pose_offsets

            if return_transl: 
                translation -= pose_offsets

            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'][..., :params['shapes'].shape[-1]]#.cuda()
            shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(params['shapes'].cuda(), (batch_size, 1, -1, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + shape_offset

            if return_transl: 
                translation += shape_offset

            posedirs = self.SMPL_NEUTRAL['posedirs']#.cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])

            if correct_Rs is not None:
                rot_mats_no_root = rot_mats[:, 1:]
                rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, joints_num-1, 3, 3)
                rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + pose_offsets

            if return_transl: 
                translation += pose_offsets

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params, rot_mats=rot_mats)

        self.s_A = A
        A = torch.matmul(bweights, self.s_A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_src_pts = can_pts + A[..., :3, 3]
        transforms = torch.matmul(A[..., :3, :3], transforms)

        if return_transl: 
            translation = torch.matmul(A[..., :3, :3], translation[..., None]).squeeze(-1) + A[..., :3, 3]

        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.matmul(smpl_src_pts, R_inv) + Th
        
        transforms = torch.matmul(R, transforms)

        if return_transl: 
            translation = torch.matmul(translation, R_inv).squeeze(-1) + Th

        # from nosmpl.vis.vis_o3d import vis_mesh_o3d
        # from smplx.body_models import SMPLX
        # import numpy as np
        # faces = self.SMPL_NEUTRAL['f'].detach().cpu().numpy().astype(np.int32)
        # vis_mesh_o3d(world_src_pts.squeeze().detach().cpu().numpy(), faces)
        # exit()

        return smpl_src_pts, world_src_pts, bweights, transforms, translation



def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs, joints_num = joints.shape[0:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

# @profile
def get_transform_params_torch(smpl, params, rot_mats=None, correct_Rs=None):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']
    betas = params['shapes']
    v_shaped = v_template[None] + torch.sum(shapedirs[None][...,:betas.shape[-1]] * betas[:,None], axis=-1).float()

    if rot_mats is None:
        # add pose blend shapes
        poses = params['poses'].reshape(-1, 3)
        # bs x 24 x 3 x 3
        rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, rot_mats.shape[1]-1, 3, 3)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] 
    Th = params['Th'] 

    return A, R, Th, joints

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat