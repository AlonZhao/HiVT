import os
import sys
# sys.path.append(os.getcwd())
# os.chdir('../')  # pwd to parent
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dlp_datamodules import DLPDataModule
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2024)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=2)#load csv batch
    parser.add_argument('--val_batch_size', type=int, default=2)#
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=2)

    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])# 训练参数 传入参数 实例化
    model = HiVT(**vars(args))# 将等同于调用：HiVT(learning_rate=0.01, num_layers=5.....) 等模型参数  实例化
    # some_function(**params)  # 等同于 some_function(a=1, b=2)
    datamodule = DLPDataModule.from_argparse_args(args)#以参数 实例化
    # trainer.fit(model, datamodule, ckpt_path='/home/alon/Learning/HiVT/lightning_logs/version_22/checkpoints/epoch=62-step=648773.ckpt')
    trainer.fit(model, datamodule) #开始调用 数据处理 loader和训练