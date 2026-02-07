import copy
import numpy as np

from typing import Any, Dict
from mmcv.parallel import DataContainer

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS#/opt/dv2-mit-bev/mmdet3d/models/builder.py这个里面创建注册器FUSIONMODELS

from .base import Base3DFusionModel

import sys, os
import importlib.util


__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()#使用FUSIONMODELS注册器的管理模块，那么注册器FUSIONMODELS就能根据字符串找到对应的类并通过.build函数创建实例
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()
        self.current_epoch = 0
        
        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        self.current_depth_weight = None
        
        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    @staticmethod
    def _unpack_dc_recursive(obj):
        """
        递归剥掉所有 DataContainer，返回纯 tensor / dict / list
        """
        if isinstance(obj, DataContainer):
            return obj.data[0]
        if isinstance(obj, list):
            return [BEVFusion._unpack_dc_recursive(item) for item in obj]
        if isinstance(obj, dict):
            return {k: BEVFusion._unpack_dc_recursive(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def _unpack_dc(self, dc):
        """
        把 list[DataContainer] 或 DataContainer → tensor
        其余类型原样返回
        """
        if isinstance(dc, list) and len(dc) > 0 and isinstance(dc[0], DataContainer):
            return torch.stack([d.data for d in dc], dim=0)
        if isinstance(dc, DataContainer):
            #print(type(dc.data)) list
            #print(len(dc.data)) 1
            #print(len(dc.data[0]))
            #for i in range(0,len(dc.data[0])):
            #    print(type(dc.data[0][i]),dc.data[0][i].shape)
            param = next(self.parameters())
            dc.data[0] = dc.data[0].to(device=param.device, dtype=param.dtype)
            return dc.data[0]
        return dc

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()#1,6,3,256,704
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
       
        if isinstance(x, list):
            x_shape = x[0].shape
        else:
            x_shape = x.shape

        x = self.encoders["camera"]["neck"](x)
        
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()#6，256，32，88

        x = x.view(B, int(BN / B), C, H, W)#B, N, C, H, W=1,6,256,32,88

        x = self.encoders["camera"]["vtransform"](#mmdet3d\models\vtransforms\depth_lss.py
            [x],
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)#[12683,5],[12683,4],[12683]
        batch_size = coords[-1, 0] + 1#1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)#mmdet3d\models\backbones\sparse_encoder.py
        return x
    

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)#tuple([12683,10,5],[12683,3],[12683])
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)#[[12683,10,5]]总共的体素数量,每个体素最多包含的点数,每个点的特征维度
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))#[[12683,4]]
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:#对每个体素内的点特征取平均
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
            
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # ===== 1. 解包（多卡兼容） =====
        img                = self._unpack_dc(self, img)
        camera2ego         = self._unpack_dc(self, camera2ego)
        lidar2ego          = self._unpack_dc(self, lidar2ego)
        lidar2camera       = self._unpack_dc(self, lidar2camera)
        lidar2image        = self._unpack_dc(self, lidar2image)
        camera_intrinsics  = self._unpack_dc(self, camera_intrinsics)
        camera2lidar       = self._unpack_dc(self, camera2lidar)
        img_aug_matrix     = self._unpack_dc(self, img_aug_matrix)
        lidar_aug_matrix   = self._unpack_dc(self, lidar_aug_matrix)
        metas              = self._unpack_dc_recursive(metas)
        
        features = []
        auxiliary_losses = {}
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":#sensor 依次为 "lidar" "camera"
                feature = self.extract_camera_features(
                    img,
                    points,#[[207998,5]]
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:#False
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)#[1,256,180,180]
                if feature!=None:
                    pass
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)#[[1,256,180,180].[1,80,180,180]]

        if not self.training:
            # avoid OOM
            features = features[::-1]#[[1,80,180,180].[1,256,180,180]]换了个位置

        if self.fuser is not None:
            x = self.fuser(features)#mmdet3d\models\fusers\conv.py  [1,256,180,180]
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)#mmdet3d\models\backbones\second.py
        x = self.decoder["neck"](x)        

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

