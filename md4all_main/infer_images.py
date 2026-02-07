import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from functools import partial
import pytorch_lightning as pl

from config.config import get_cfg
from trainer import Md4All
from visualization.visualize import Visualizer


class Crop:
    """图像裁剪变换类"""
    def __init__(self, top, left, height, width):
        self.crop = partial(transforms.functional.crop, top=top, left=left, height=height, width=width)

    def __call__(self, img):
        return self.crop(img)


def get_transforms(cfg):
    """获取图像预处理变换"""
    transform = transforms.Compose([
        Crop(cfg.DATASET.AUGMENTATION.CROP.TOP, 
             cfg.DATASET.AUGMENTATION.CROP.LEFT, 
             cfg.DATASET.AUGMENTATION.CROP.HEIGHT, 
             cfg.DATASET.AUGMENTATION.CROP.WIDTH),
        transforms.Resize(size=(cfg.DATASET.AUGMENTATION.RESIZE.HEIGHT, 
                               cfg.DATASET.AUGMENTATION.RESIZE.WIDTH),
                         interpolation=transforms.InterpolationMode.LANCZOS, 
                         antialias=True),
        transforms.ToTensor()
    ])
    return transform


def prepare_batch(images, transform, daytimes=None, filenames=None):
    """准备模型输入批次
    
    Args:
        images: 图像列表，可以是PIL.Image对象或numpy数组
        transform: 图像预处理变换
        daytimes: 可选，图像对应的时间信息列表
        filenames: 可选，图像文件名列表，用于保存结果
    
    Returns:
        处理后的批次数据字典
    """
    batch_size = len(images)
    processed_images = []
    
    # 处理每张图像
    for i, img in enumerate(images):
        # 确保图像是PIL.Image格式
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise TypeError(f"图像应为PIL.Image或numpy数组，实际类型: {type(img)}")
        
        # 应用预处理
        processed_img = transform(img)
        processed_images.append(processed_img)
    
    # 创建批次字典
    batch = {
        ('color', 0): torch.stack(processed_images)
    }
    
    # 添加文件名信息
    if filenames is None:
        filenames = [f"image_{i}" for i in range(batch_size)]
    batch[('filename', 0)] = filenames
    
    # 添加时间信息（如果提供）
    if daytimes:
        if len(daytimes) != batch_size:
            raise ValueError(f"daytimes长度应与图像数量匹配，实际: {len(daytimes)} vs {batch_size}")
        batch['weather'] = daytimes
    
    return batch


def infer_images(images, config_path, output_path=None, daytimes=None, filenames=None, device='cuda'):
    """直接使用图像数据进行深度估计推理
    
    Args:
        images: 图像列表，可以是PIL.Image对象或numpy数组，支持同时处理多张图像
        config_path: 配置文件路径
        output_path: 可选，输出结果保存路径
        daytimes: 可选，图像对应的时间信息列表
        filenames: 可选，图像文件名列表，用于保存结果
        device: 运行设备，默认为'cuda'
    
    Returns:
        深度估计结果列表
    """
    # 加载配置
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    
    # 设置输出路径
    if output_path:
        cfg.EVALUATION.SAVE.QUANTITATIVE_RES_PATH = output_path
        cfg.EVALUATION.SAVE.QUALITATIVE_RES_PATH = output_path
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
    
    # 加载模型
    if cfg.LOAD.CHECKPOINT_PATH is None:
        raise AssertionError("配置文件中需要指定CHECKPOINT_PATH")
    
    # 加载模型并移至指定设备
    model = Md4All.load_from_checkpoint(cfg.LOAD.CHECKPOINT_PATH, cfg=cfg, is_train=False)
    model.to(device)
    model.eval()
    
    # 创建可视化器
    if output_path:
        visualizer = Visualizer(
            model.temp_context, 
            False,  # 不使用光度损失可视化
            cfg.EVALUATION.SAVE.QUALITATIVE_RES_PATH, 
            cfg.EVALUATION.SAVE.VISUALIZATION_SET,
            cfg.EVALUATION.SAVE.RGB, 
            cfg.EVALUATION.SAVE.DEPTH.PRED,
            cfg.EVALUATION.SAVE.DEPTH.GT,
            cfg.EVALUATION.DEPTH.MIN_DEPTH,
            cfg.EVALUATION.DEPTH.MAX_DEPTH
        )
    else:
        visualizer = None
    
    # 获取图像预处理变换
    transform = get_transforms(cfg)
    
    # 准备批次数据
    batch = prepare_batch(images, transform, daytimes, filenames)
    
    # 将批次数据移至指定设备
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # 执行推理
    with torch.no_grad():
        outputs = model.depth_model(batch[('color', 0)], batch.get('weather'))
    
    # 获取深度图
    depth_maps = []
    for i in range(len(images)):
        # 转换为实际深度值
        depth = model.evaluator.inv2depth(outputs[('disp', 0, 0)][i]).cpu().numpy()
        depth_maps.append(depth)
    
    # 保存结果（如果指定了输出路径）
    if output_path:
        for i, depth in enumerate(depth_maps):
            filename = batch[('filename', 0)][i]
            # 保存深度图为npy文件
            np.save(os.path.join(output_path, f"{filename}_depth.npy"), depth)
        
        # 保存可视化结果
        if visualizer:
            visualizer.visualize_predict(batch, outputs)
    
    return depth_maps


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="直接使用图像数据进行深度估计推理")
    parser.add_argument("--config", type=str, help="配置文件路径", required=True)
    parser.add_argument("--image_paths", type=str, nargs='+', help="图像文件路径列表", required=True)
    parser.add_argument("--output_path", type=str, help="输出结果保存路径")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    
    args = parser.parse_args()
    
    # 加载图像
    images = [Image.open(img_path) for img_path in args.image_paths]
    filenames = [os.path.basename(path).split('.')[0] for path in args.image_paths]
    
    # 执行推理
    depth_maps = infer_images(
        images=images,
        config_path=args.config,
        output_path=args.output_path,
        filenames=filenames,
        device=args.device
    )
    
    print(f"成功处理 {len(depth_maps)} 张图像")
