from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from dlp_datasets import DLPDataset

class DLPDataModule(LightningDataModule):#数据集接口 数据加载和预处理

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,# true 加速数据转移过程
                 persistent_workers: bool = True, # true 数据加载进程保持
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        super(DLPDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius # local region的半径

    def prepare_data(self) -> None: #在训练前调用 prepare_data，并在每个训练进程中调用 setup
        print('prepare_data:')
        DLPDataset(self.root, 'train', self.train_transform, self.local_radius)
        DLPDataset(self.root, 'val', self.val_transform, self.local_radius)
        

    def setup(self, stage: Optional[str] = None) -> None: #分训练和验证 确保 setup 方法能够兼容不同版本的 PyTorch Lightning，同时保持灵活性。
        print('setup : ')
        self.train_dataset = DLPDataset(self.root, 'train', self.train_transform, self.local_radius)
        self.val_dataset = DLPDataset(self.root, 'val', self.val_transform, self.local_radius)

    def train_dataloader(self): # 加载训练数据部分 fit 时候自动调用
        print('train_dataloader:')
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self): # 加载训练数据部分 每一轮结束 时候自动调用
        print('val_dataloader:')
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
