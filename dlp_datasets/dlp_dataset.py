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
from itertools import permutations
from itertools import product
from typing import  Dict
import sys
# sys.path.append(os.getcwd())
# os.chdir('../')  # pwd to parent
import numpy as np
import pandas as pd
import glob
from utils import TemporalData

class DLPDataset(Dataset):# 注意是 geo中的

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius 
        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        else:
            raise ValueError(split + ' is not valid')# 设置数据存储目录 设置文件名字
        # self.root = root
        print('train or val : ', split)
        self.root = root # .../dlp-root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = os.listdir(self.processed_dir)# explore to get instead of splitext
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]# 组合后 定义路径
        super(DLPDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:# 原始数据存储目录 + val + data 
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:# 原始数据存储目录 + val + processed 
        return os.path.join(self.root, self._directory, 'processed')
        

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:#返回原始文件列表
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def len(self) -> int: # required
        return len(self._processed_file_names)

    def get(self, idx) -> Data: # required
        return torch.load(self.processed_paths[idx])
    
    def process(self) -> None:# execute  at  first processing  
        print('process csv files')
        csv_files = list_csv_files_sorted(self.raw_dir)# 调通后验证这一块是否有点多余了 batch必须是偶数 因为要一次性取出一对儿
        # 每次读取两个文件
        print('pre-checking')
        for i in tqdm(range(0, len(csv_files), 2)):
            lanes_path = csv_files[i]
            obs_path = csv_files[i+1]
            lane_filename = os.path.basename(lanes_path)
            obs_filename = os.path.basename(obs_path)
             ##增加了文件名字配对检查
            lane_date = lane_filename.split('_lines')[0]
            obs_date = obs_filename.split('_objs')[0]

            if not (lane_date==obs_date):
                print("Not pair!")
                print(lane_date)
                print(obs_date)
                return 


        for i in tqdm(range(0, len(csv_files), 2)):
            lanes_path = csv_files[i]
            obs_path = csv_files[i+1]

   
            ##增加了文件大小判断
            lanes_file_size = os.path.getsize(lanes_path) / 1024
            obs_file_size = os.path.getsize(obs_path) / 1024
            if obs_file_size < 50 or  lanes_file_size < 500:
                continue
            lane_filename = os.path.basename(lanes_path)
            obs_filename = os.path.basename(obs_path)
            # 查找文件名前缀的公共部分
            prefix = os.path.commonprefix([lane_filename, obs_filename])
            # print(prefix)
            # 每个文件中取得的帧数
            obs_df = pd.read_csv(obs_path)  
            timestamps = list(np.sort(obs_df['frame_idx'].unique()))
            # 确认起始终止frameid 向后保留100s 向前保留20s
            frame_start = 19
            frame_end = len(timestamps) - 51
            # 间隔20frame生成数据 也就是2s
            for frame_to_get in range(frame_start,frame_end,10):
                # 筛选掉当前没有障碍物的数据
                df_19 = obs_df[obs_df['frame_idx'] == timestamps[frame_to_get]]# 
                df_18 = obs_df[obs_df['frame_idx'] == timestamps[frame_to_get - 1]]# 
                df_20 = obs_df[obs_df['frame_idx'] == timestamps[frame_to_get + 1]]# 

                actors_18 = [actor_id for actor_id in df_18['track_id']]# 当前时刻所有actors
                actors_19 = [actor_id for actor_id in df_19['track_id']]# 当前时刻所有actors
                actors_20 = [actor_id for actor_id in df_20['track_id']]# 当前时刻所有actors

                if len(actors_19) < 2:
                    continue

                set_18 = {id for id in actors_18 if id > 0}
                set_19 = {id for id in actors_19 if id > 0}
                set_20 = {id for id in actors_20 if id > 0}

                common_ids = set_18 & set_19 & set_20
                if not common_ids:
                    continue

                kwargs = get_features(raw_lane_path=lanes_path,raw_path_obs = obs_path,frame_to_get = frame_to_get)
                _data = TemporalData(**kwargs)#封装成自定义数据类型
               
                file_to_save_path = os.path.join(self.processed_dir, prefix + str(frame_to_get) + '.pt')
                # print(f"Saving file to: {file_to_save_path}")
                torch.save(_data, file_to_save_path)
                # try:
                #     torch.save(_data, file_to_save_path)
                # except Exception as e:
                #     print(f"Error saving file: {e}")
                # torch.save(_data, os.path.join(self.processed_dir, f"{prefix}_{frame_to_get}.pt"))
                # torch.save(_data, os.path.join(self.processed_dir, prefix + str(frame_to_get) + '.pt'))# 、
def get_features( raw_lane_path: str,
              raw_path_obs: str,
              frame_to_get: int
              )-> Dict:
    file_name = os.path.basename(raw_path_obs)
    name_parts = file_name.split('_no')[0]
# obs feature extracttion
    obs_df = pd.read_csv(raw_path_obs)  
    timestamps = list(np.sort(obs_df['frame_idx'].unique()))
    if( len(timestamps) < 70): # 5s prediction 50->70
        # print('Not Enough Frames!')
        return
    if(frame_to_get<19):
        # print('Not Enough History!')
        return
    if(frame_to_get > len(timestamps) -51):  # 5s prediction 31->51
        # print('Not Enough Future!')
        return 
    

### filting 50 frames around frame_to_get
    timestamps = timestamps[ frame_to_get - 19: frame_to_get + 51]
    # print('from to: ',timestamps[0],timestamps[-1]) ## debug message
### re-filting obs df in 50 sampled frames
    obs_df = obs_df[obs_df['frame_idx'].isin(timestamps)]
### history part
    historical_timestamps = timestamps[: 20]
    historical_obs_df = obs_df[obs_df['frame_idx'].isin(historical_timestamps)] 

###  actor track id in history time timestamp  
    actor_ids = list(historical_obs_df['track_id'].unique())#历史内所有目标的序列ID 是一个包含所有目标 ID 的列表。
### filted from 50 frames frame里面都是历史出现过的ID 包含了历史和可能得未来
    obs_df = obs_df[obs_df['track_id'].isin(actor_ids)] #保留历史出现过的ID即可
    
    actor_num = len(actor_ids)
    # print('actors ids :',actor_ids)

    av_df = obs_df[obs_df['object_type'] == 1].iloc #aV frameS
    av_index = actor_ids.index(av_df[0]['track_id'])#av  index in actor_ids 
    agent_index = 0 # nonsense here, just 0

    # ready to make the scene centered at AV at now(19) moment
    origin = torch.tensor([av_df[19]['rel_x'], av_df[19]['rel_y']], dtype=torch.float)
    # print('origin, av boost pos:', origin)
    av_heading_vector = origin - torch.tensor([av_df[18]['rel_x'], av_df[18]['rel_y']], dtype=torch.float)# attetion head wrt. boost
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    # print('av theta in boost at 19 stamps ',theta)
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])
    
### initialization features in .pt
    x = torch.zeros(actor_num, 70, 2, dtype=torch.float) # 5s prediction 
    edge_index = torch.LongTensor(list(permutations(range(actor_num), 2))).t().contiguous()#(2, N * N-1) Ai-Aj interaction
    padding_mask = torch.ones(actor_num, 70, dtype=torch.bool) # 5s prediction 
    bos_mask = torch.zeros(actor_num, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(actor_num, dtype=torch.float)
    # full frames samples
    complete_samples=[] 

### processing pos by each actor
    #an actor_id with a group of actor_df
    for actor_id, actor_df in obs_df.groupby('track_id'): 
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['frame_idx']]# 在最原始文件中的 时间戳的位置
        # print('actor_id : ',actor_id)
        # print('node_idx : ',node_idx)
        # print('node_steps : ',node_steps)
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  
            padding_mask[node_idx, 20:] = True 
        #pos of the selected actor
        xy = torch.from_numpy(np.stack([actor_df['rel_x'].values, actor_df['rel_y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin , rotate_mat)# center with the AV at 19 s
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
        # mark full sample
        # if len(node_historical_steps) == 20:
            # print('full node_historical_steps stamps:', node_idx)
        if len(node_steps) == 70: # 7s prediction 
            # print('full _steps stamps:', node_idx)
            complete_samples.append(node_idx)
        
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]] # actor heading 
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True
    # print('complete actor index (not id):',complete_samples)
    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]# nonsense
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20] #
    positions = x.clone()#差分前， 保留原始轨迹位置，以AV为中心

### differential vector

    # gt future wrt. x(19)
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(actor_num, 50, 2),# 5s prediction 
                            x[:, 20:] - x[:, 19].unsqueeze(-2))#torch.where(condition, x_if_true, x_if_false)
    # past wrt. past.shift(-1)
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(actor_num, 19, 2),
                              x[:, 1: 20] - x[:, : 19])
    x[:, 0] = torch.zeros(actor_num, 2)

    # get lane features at the current time step
    df_19 = obs_df[obs_df['frame_idx'] == timestamps[19]]# 
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['track_id']]# 当前时刻所有actors
    node_positions_19 = torch.from_numpy(np.stack([df_19['rel_x'].values, df_19['rel_y'].values], axis=-1)).float()
    node_positions_rel = torch.matmul(node_positions_19 - origin, rotate_mat).float()
    # gt
    y = x[:, 20:] 

### lanes info file 
    lanes_df = pd.read_csv(raw_lane_path)
    num_rows = lanes_df.shape[0]
    num_columns = lanes_df.shape[1]
    target_lanes_df =   lanes_df[lanes_df['frame_idx'] ==   timestamps[19] ].iloc

    lane_idx_start =  3 # depends on file format
    # select lanes points  by x y respt.

    #原来的程序
    # x_pos = target_lanes_df[:, range(lane_idx_start, num_columns, 2)]
    # y_pos = target_lanes_df[:, range(lane_idx_start + 1 , num_columns, 2)]
    #现在的程序 默认100个点
    x_pos = target_lanes_df[:, range(lane_idx_start, num_columns  , 2)]
    y_pos = target_lanes_df[:, range(lane_idx_start + 1 , num_columns  , 2)]
    x_pos = x_pos.dropna(axis=1)
    y_pos = y_pos.dropna(axis=1)

### re-organize points by numpy
    x_pos_np = x_pos.to_numpy() # m * n
    y_pos_np = y_pos.to_numpy()
    # print('first 5 x lanes pos: ')
    # print((x_pos_np[:,:5]))
    # print('first 5 y lanes pos: ')
    # print((y_pos_np[:,:5]))

    #################### before difference, get raw lane points to convert to boost and  draw after inteference #########################
    raw_one_x = np.hstack(x_pos_np)# flatten multi-lanes' x to one dimension 
    raw_one_y = np.hstack(y_pos_np)# 1 * m*n
    raw_lane_pos  = np.vstack((raw_one_x, raw_one_y))## combine 2 * (m*n)
    raw_lane_pos = torch.from_numpy(raw_lane_pos).float()
    # convert to boost coordinate
    raw_lane_pos = raw_lane_pos.permute(1,0)#  (m*n) * 2
    inv_rotate_mat = rotate_mat.t()
    raw_lane_pos[:] = torch.matmul(raw_lane_pos[:],inv_rotate_mat) + origin
    #################### before difference, get raw lane points to store and draw after inteference #########################


    #################### before concatenate , get each lane's diff vector  to store  #########################
    vector_x_pos_np = x_pos_np[:,1:] - x_pos_np[:,:-1] 
    vector_y_pos_np = y_pos_np[:,1:] - y_pos_np[:,:-1]# m * (n-1)
    one_vx = np.hstack(vector_x_pos_np) # # flatten multi-lanes' vector x to one dimension 
    one_vy = np.hstack(vector_y_pos_np)
    lane_vector_np  = np.vstack((one_vx, one_vy)) #combine 2 * (m * (n-1) )

    lane_vectors = torch.from_numpy(lane_vector_np).float()
    lane_vectors = lane_vectors.permute(1,0) #   (m * (n-1) ) * 2
    # print('lane_vectors.shape is ', lane_vectors.shape)
    #################### before concatenate , get each lane's diff vector  to store  #########################


    head_raw_one_x = np.hstack(x_pos_np[:,:-1])## regard the vector start point as lanes_position, dimension is as the vec
    head_raw_one_y = np.hstack(y_pos_np[:,:-1])
    lanes_position_np  = np.vstack((head_raw_one_x, head_raw_one_y))
    lanes_position = torch.from_numpy(lanes_position_np).float()
    lanes_position = lanes_position.permute(1,0)# 2 * (m * (n-1) )
    
### get lanes map for all actors
    ones_tensor = torch.ones(lane_vectors.shape[0])
    is_intersections = ones_tensor == 0 # default false
    turn_directions = torch.zeros(lane_vectors.shape[0]) # none
    traffic_controls =  ones_tensor == 0 # default false
    #AL interaction
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds_19))).t().contiguous()# diff data type
    #AL vectors at every point to every actor 
    lane_actor_vectors = \
        lanes_position.repeat_interleave(len(node_inds_19), dim=0) - node_positions_rel.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < 100 # 原来是50 建议改成100 当前处理的尚为50
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]
    seq_id = frame_to_get
# L = (m * (n-1) )
    return  {
        'x': x[:, : 20],  # [N, 20, 2] 
        'positions': positions,  # [N, 70, 2]   
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 50, 2]
        'num_nodes': actor_num,
        'padding_mask': padding_mask,  # [N, 70]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': 'Hefei',
        'origin': origin.unsqueeze(0),
        'theta': theta,
        'raw_lane_pos': raw_lane_pos, #n*2
        'complete_samples': complete_samples,
        'filenamne':  f"{name_parts}_{seq_id}"
    }

def list_csv_files_sorted(folder_path):
    # 查找所有csv文件的路径并按照字母顺序排序
    # 同一时刻 现有line后有obs
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    # 创建一个列表存储文件名
    file_list = [file for file in csv_files]

    return file_list