#!/bin/bash

# 使用nuScenes mini版本运行md4all深度评估
echo "开始使用nuScenes mini版本运行评估..."
python evaluation/evaluate_depth.py --config /opt/dv2-mit-bev/md4all-main/config/eval_md4allDDa_80m_nuscenes_mini_val.yaml
echo "评估完成！结果保存在：/opt/dv2-mit-bev/md4all-main/results/quantitative/nuscenes/mini_val"