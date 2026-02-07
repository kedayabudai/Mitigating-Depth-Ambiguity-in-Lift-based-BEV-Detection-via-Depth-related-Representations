# weight_transfer_smart.py
import torch
import torch.nn as nn
import logging

def smart_transfer_lss_to_sappmlss(old_state_dict, new_model, logger=None):
    """
    智能权重迁移：从LSSFPN迁移到SAPPMLSSFPN
    """
    new_state_dict = new_model.state_dict()
    
    # 映射规则：旧键名 -> 新键名
    key_mapping = {
        # fuse模块完全一致
        'fuse.0.weight': 'fuse.0.weight',
        'fuse.0.bias': 'fuse.0.bias',
        'fuse.1.weight': 'fuse.1.weight',
        'fuse.1.bias': 'fuse.1.bias',
        'fuse.1.running_mean': 'fuse.1.running_mean',
        'fuse.1.running_var': 'fuse.1.running_var',
        'fuse.1.num_batches_tracked': 'fuse.1.num_batches_tracked',
        'fuse.3.weight': 'fuse.3.weight',
        'fuse.3.bias': 'fuse.3.bias',
        'fuse.4.weight': 'fuse.4.weight',
        'fuse.4.bias': 'fuse.4.bias',
        'fuse.4.running_mean': 'fuse.4.running_mean',
        'fuse.4.running_var': 'fuse.4.running_var',
        'fuse.4.num_batches_tracked': 'fuse.4.num_batches_tracked',
        
        # upsample模块
        'upsample.1.weight': 'upsample.1.weight',
        'upsample.1.bias': 'upsample.1.bias',
        'upsample.2.weight': 'upsample.2.weight',
        'upsample.2.bias': 'upsample.2.bias',
        'upsample.2.running_mean': 'upsample.2.running_mean',
        'upsample.2.running_var': 'upsample.2.running_var',
        'upsample.2.num_batches_tracked': 'upsample.2.num_batches_tracked',
    }
    
    transferred_keys = []
    random_init_keys = []
    
    # 1. 迁移所有匹配的权重
    for old_key, new_key in key_mapping.items():
        full_old_key = f'decoder.neck.{old_key}'
        full_new_key = f'decoder.neck.{new_key}'
        
        if full_old_key in old_state_dict and full_new_key in new_state_dict:
            old_weight = old_state_dict[full_old_key]
            new_weight = new_state_dict[full_new_key]
            
            if old_weight.shape == new_weight.shape:
                new_state_dict[full_new_key] = old_weight.clone()
                transferred_keys.append(full_new_key)
                
                if logger:
                    logger.info(f"✓ Transferred: {full_old_key} -> {full_new_key}")
    
    # 2. SAPPM模块使用特殊的初始化策略
    # SAPPM模块需要从现有Conv层借用初始化
    if hasattr(new_model.decoder.neck, 'sapm'):
        # 获取一个已迁移的卷积层权重作为参考
        reference_conv_key = 'decoder.neck.fuse.0.weight'
        if reference_conv_key in new_state_dict:
            reference_weight = new_state_dict[reference_conv_key]
            
            # 初始化SAPPM的卷积层
            for name, param in new_model.decoder.neck.sapm.named_parameters():
                full_name = f'decoder.neck.sapm.{name}'
                if 'weight' in name and param.dim() == 4:  # 卷积权重
                    # 使用He初始化，但参考已有权重的方差
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    random_init_keys.append(full_name)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                    random_init_keys.append(full_name)
                    
                if logger:
                    logger.info(f"✧ Initialized: {full_name}")
    
    # 3. 加载迁移后的权重
    missing_keys, unexpected_keys = new_model.load_state_dict(new_state_dict, strict=False)
    
    # 4. 打印统计信息
    if logger:
        logger.info("\n" + "="*60)
        logger.info("WEIGHT TRANSFER SUMMARY")
        logger.info("="*60)
        logger.info(f"Total transferred keys: {len(transferred_keys)}")
        logger.info(f"Total randomly initialized keys: {len(random_init_keys)}")
        logger.info(f"Missing keys (expected): {len(missing_keys)}")
        logger.info(f"Unexpected keys (ignored): {len(unexpected_keys)}")
        
        if missing_keys:
            logger.info("\nMissing keys (SAPPM模块预期中的):")
            for key in missing_keys[:10]:  # 只显示前10个
                if 'sapm' in key:
                    logger.info(f"  {key}")
        
        if transferred_keys:
            logger.info("\nExample transferred keys:")
            for key in transferred_keys[:5]:
                logger.info(f"  {key}")
    
    return new_model, transferred_keys