import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from models.hivt import HiVT
from utils import TemporalData
import os 
import random

model_unit = HiVT.load_from_checkpoint('/home/alon/Learning/HiVT/lightning_logs/version_26/checkpoints/epoch=63-step=659071.ckpt'
                                       , parallel=True)
_info = model_unit.eval()
file_path = '/home/alon/Learning/hivt_data/val/processed/20099.pt'
# 加载文件内容
test_data = torch.load(file_path)
# 执行所需的操作，例如打印或处理数据
print(f"Loaded data from {file_path}")
print('agent_index=',test_data['agent_index'])
print('av_index=',test_data['av_index'])
with torch.no_grad(): 
    y_hat_agent, pi_agent, seq_id ,position_sce= model_unit.predition_unit_batch(test_data,batch_idx=0)
y_hat_agent = y_hat_agent.permute(0, 1, 2, 3)
print('y_hat_agent.size = ',y_hat_agent.size()) #[29, 6, 30, 2]
# print('pi_agent:',pi_agent)
pred_traj_np = y_hat_agent.cpu().numpy()
full_traj = position_sce
if full_traj.is_cuda:
    full_traj = full_traj.cpu()
# 转换为 NumPy 数组
full_traj_np = full_traj.numpy()

index_agent = test_data['agent_index']
av_agent = test_data['av_index']

