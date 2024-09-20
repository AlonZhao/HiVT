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
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData


class ArgoverseV1Dataset(Dataset):# 注意是 geo中的metric

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
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
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]#分割为文件名和扩展名 定义文件名字
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]# 组合后 定义路径
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:# 原始数据存储目录
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:# 指定处理数据存储目录
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

    def process(self) -> None:# 初次处理数据才会调用 
        print('process alon')
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            data = TemporalData(**kwargs)#封装成自定义数据类型
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))# 根据目录存为pt文件

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    df = pd.read_csv(raw_path)

    # filter out actors that are unseen during the historical time steps
    # all-timestamp -> history  timestamp 筛选 
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)] 
    # history time timestamp -> 筛选ID种类 
    actor_ids = list(historical_df['TRACK_ID'].unique())#历史内所有目标的序列ID 是一个包含所有目标 ID 的列表。
    #回到全frame 用历史ID种类  筛选整个frame 保证  frame里面都是历史出现过的ID 包含了历史和未来
    df = df[df['TRACK_ID'].isin(actor_ids)] #保留历史出现过的ID
    num_nodes = len(actor_ids)# 历史中有的ID

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc #自动驾驶车的 frame
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])#av_df 中第一个 TRACK_ID 在 actor_ids 列表中的索引位置。
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc  #预测目标
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]#整个文件都是一样的

    # ready to make the scene centered at AV
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)#
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])#实际上不需要  local region的 agent centric
    # 旋转不会影响HiVT的性能

    # initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()#(2, num_edges)且连续的有向边
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):# 以ID为中心出发更新 将 DataFrame 按照 TRACK_ID 列进行分组，每个分组包含一个 actor_id 和相应的数据actor_df
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]# 在最原始文件中的 时间戳的位置
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True #如果在时间步 19 的掩码仍然是 True，则表示该节点在时间步 19 之后的所有时间步都没有数据，因此将这些时间步的掩码设置为 True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float() #当前的actor的轨迹点序列 沿着时间向下延展
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)#以AV为中心的变换 该actor 所有轨迹点序列
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]# wrt. AV
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]# begining-of-Sequence Mask
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20] # 某个时间t试试是有效的并且之前的时间是无效的 标记那些在时间步 t 有数据而时间步 t-1 需要填充的时间步
    positions = x.clone()#保留原始轨迹位置，以AV为中心
    #使用偏差
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))# 未来时刻-当前时刻的偏差 torch.where(condition, x_if_true, x_if_false)
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19])#1-19状态   0-18状态
    x[:, 0] = torch.zeros(num_nodes, 2)#将第一个时间步的值设置为零

    # get lane features at the current time step
    df_19 = df[df['TIMESTAMP'] == timestamps[19]]# 
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]# 当前时刻所有actors
    node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()
    #获取特征 向量，是否交叉，转弯
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius)

    y = None if split == 'test' else x[:, 20:] # 轨迹gt 但是是处理过的 AV中心 而且差分
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]#分离文件名和扩展名。返回一个元组 取得文件名字

    return {
        'x': x[:, : 20],  # [N, 20, 2] processed transformed differential history trajectory
        'positions': positions,  # [N, 50, 2]   comlete trajectory after transformed but before differential
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
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
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def get_lane_features(am: ArgoverseMap,# api地图
                      node_inds: List[int], # 当前时刻所有agent的索引列表
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,#AV的
                      rotate_mat: torch.Tensor,#AV的 
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:#收集范围内的lane id 一个城市是唯一指定的
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()# 以自车为坐标系 原点和方向
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])#计算lane Vector
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)
    # 所有可能车道和agents 的pair
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius# 50m 以外的不关心
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
