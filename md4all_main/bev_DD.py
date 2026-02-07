import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from config.config import get_cfg
from data.custom_dataset import CustomDataModule
from trainer import Md4All

from utils.depth import inv2depth

def init_model():
    config_path = "/bevfusion-main/md4all_main/config/bev_DD.yaml"  # 这里可以设置你想要的默认配置文件路径
    
    # 加载配置
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    torch.backends.cudnn.benchmark = cfg.SYSTEM.BENCHMARK

    # Check if checkpoint path is specified
    if cfg.LOAD.CHECKPOINT_PATH is None:
        raise AssertionError("For prediction you need to specify a path to a checkpoint")
    
    # Load model checkpoint here manually as it does not work along with trainer.predict
    model = Md4All.load_from_checkpoint(cfg.LOAD.CHECKPOINT_PATH, cfg=cfg, is_train=False)
    return model

def infer_depth(model,input):
    H, W = input.shape[2], input.shape[3]
    img = F.interpolate(
        input=input,
        size=(320, 576),  # 目标空间维度（H, W）
        mode='bilinear',  # 插值模式：bilinear（双线性插值，适合图像）/ nearest（最近邻）/ bicubic（双三次）
        align_corners=False  # 避免边缘畸变，CV任务默认False
        )
    outputs = model.depth_model(img, None)#[bitch,3,320,576]
    #depths = inv2depth(outputs[('disp', 0, 0)]).cpu().numpy()

    depths = inv2depth(outputs[('disp', 0, 0)])
    depths = F.interpolate(
        input=depths,
        size=(H, W),  # 目标空间维度（H, W）
        mode='bilinear',  # 插值模式：bilinear（双线性插值，适合图像）/ nearest（最近邻）/ bicubic（双三次）
        align_corners=False  # 避免边缘畸变，CV任务默认False
        )
    return depths

