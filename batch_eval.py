# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import pytorch_lightning as pl

from torch_geometric.data import DataLoader # 之前常用的torch.utils.data.DataLoader 跟pl不兼容
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from typing import Optional
from argparse import ArgumentParser
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
from tqdm import tqdm
from models.hivt import HiVT

class DLPDataset(Dataset):# 注意是 geo中的metric

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')# 设置数据存储目录 设置文件名字
        # self.root = root
        self.root = '/home/alon/Learning/HiVT'
        # self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = os.listdir(self.processed_dir)
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]# 组合后 定义路径
        super(DLPDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:# 原始数据存储目录 + val + data 
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:# 原始数据存储目录 + val + processed 
        # return os.path.join(self.root, self._directory, 'pre-process')
        return os.path.join(self.root,  'pre-process')

    # @property
    # def raw_file_names(self) -> Union[str, List[str], Tuple]:#返回原始文件列表
    #     return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:# 初次处理数据才会调用 
        print('process alon')

    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])



if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    val_dataset = DLPDataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)
