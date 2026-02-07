import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
import torch.distributed as dist

from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
from torch.nn.parallel import DistributedDataParallel
import torch.cuda


def apply_phase1_freeze(model, logger, rank):
    """
    第一阶段冻结策略：只训练SAPPMLSSFPN，其他模块全部冻结
    所有rank必须执行相同的操作！
    """
    if rank == 0:
        logger.info("=" * 60)
        logger.info("PHASE 1: Freezing all modules except SAPPMLSSFPN")
        logger.info("=" * 60)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    
    # 第一阶段：冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # 只解冻SAPPMLSSFPN（decoder.neck）
    trainable_params = 0
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'neck'):
        for name, param in model.decoder.neck.named_parameters():
            param.requires_grad = True
            trainable_params += param.numel()
    
    # 固定DPTHead的参数（如果存在）
    if hasattr(model, 'dpt_head') and model.dpt_head is not None:
        for param in model.dpt_head.parameters():
            param.requires_grad = False
        if rank == 0:
            logger.info("✓ DPTHead parameters frozen")
    
    # 打印统计信息
    if rank == 0:
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params*100:.4f}%")
    
    return model

def main():

    # ---------------------------
    # 1. 初始化 torchrun 分布式环境
    # ---------------------------
    dist.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    # ---------------------------
    # 2. 设置当前 GPU (torchrun 会自动提供 local_rank)
    # ---------------------------
    # torchrun 自动设置 LOCAL_RANK 环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # ---------------------------
    # 3. 运行目录
    # ---------------------------
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # 保存 config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # ---------------------------
    # 4. 日志
    # 只有 rank0 创建日志文件
    # ---------------------------
    rank = dist.get_rank()
    log_file = None
    if rank == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")

    logger = get_root_logger(log_file=log_file)

    if rank == 0:
        logger.info(f"Config:\n{cfg.pretty_text}")

    # ---------------------------
    # 5. 随机种子
    # ---------------------------
    if cfg.seed is not None:
        if rank == 0:
            logger.info(f"Set random seed to {cfg.seed}, deterministic: {cfg.deterministic}")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ---------------------------
    # 6. 构建数据集
    # ---------------------------
    datasets = [build_dataset(cfg.data.train)]

    # ---------------------------
    # 7. 构建模型
    # ---------------------------
    model = build_model(cfg.model)
    model = apply_phase1_freeze(model, logger, rank)

    # SyncBN
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    if rank == 0:
        logger.info(f"Model:\n{model}")

    # ---------------------------
    # 8. 开始训练
    # ---------------------------
    # 固定DPTHead的参数，不进行更新
    # 这样可以允许梯度从DPTHead输出流向LSS模型，同时保持DPTHead参数不变
    if hasattr(model, 'dpt_head') and model.dpt_head is not None:
        for param in model.dpt_head.parameters():
            param.requires_grad = False
        logger.info("DPTHead parameters have been frozen")
        
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime()) if rank == 0 else "",
    )


if __name__ == "__main__":
    main()