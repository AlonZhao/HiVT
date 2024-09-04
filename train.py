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
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import ArgoverseV1DataModule
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2024)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)# 可选参数
    parser.add_argument('--train_batch_size', type=int, default=20)##改动适应4070 8G显存
    parser.add_argument('--val_batch_size', type=int, default=20)##改动适应4070 8G显存
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)

    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])# 训练参数 传入参数 实例化
    model = HiVT(**vars(args))# 将等同于调用：HiVT(learning_rate=0.01, num_layers=5.....) 等模型参数  实例化
    # some_function(**params)  # 等同于 some_function(a=1, b=2)
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)#以参数 实例化
    trainer.fit(model, datamodule, ckpt_path='/home/alon/Learning/HiVT/lightning_logs/version_22/checkpoints/epoch=62-step=648773.ckpt')
    # trainer.fit(model, datamodule) #开始调用 数据处理 loader和训练