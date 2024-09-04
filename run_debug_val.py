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
    model_unit.validation_step(test_data,batch_idx=0)


