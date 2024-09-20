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
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder
from utils import TemporalData


class HiVT(pl.LightningModule):#继承

    def __init__(self,
                 historical_steps: int,# 过去时间步长20
                 future_steps: int,# 30
                 num_modes: int,# 6个输出轨迹
                 rotate: bool,# 
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,#
                 num_heads: int,#
                 dropout: float,
                 num_temporal_layers: int,#子模块层次数量
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool, #并行 agent-agent可以并行计算
                 lr: float,#优化器参数
                 weight_decay: float,
                 T_max: int,
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        # 三大模型 
        ## 获取预测目标及其车道的相关性信息 对称性编码 各自中心
        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        ## 全局信息传递 
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        #获取信息后的解码
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        # 两个损失函数
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()
        # 前向传播
    def forward(self, data: TemporalData):
        # print("Forward is called", flush=True)
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)# y: wrt.AV  row vector from AV to agent
                # print('data.y = torch.bmm(data.y, rotate_mat)')
                
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return y_hat, pi

    def training_step(self, data, batch_idx):
        y_hat, pi = self(data) #  [F, N, H, 4], [N, F]
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]# 有效未来时刻轨迹 [N, 30]true false
        valid_steps = reg_mask.sum(dim=-1) # 有效未来时刻总数 N
        cls_mask = valid_steps > 0 #识别哪些agent至少有一个有效的未来时间步 valid_steps[cls_mask] 选出来 N的 true / false 
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N, H, 2] [ N, H, 2]-> [F, N] 因为同一个agent有效点一样多 所以比较总合就行
        best_mode = l2_norm.argmin(dim=0) # [N]
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]# 最好的模态是 总的误差最小的 [F,N,H,4] -> [N,H,4]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])# 最佳模态[有效未来时刻]的 Laplace NLL [sum Hi,4] 会混合在一起 所有agent的最好的模态
        #[F, 有效代理数量]维度的误差  /  [有效代理数量]的各总步数   在不同agent维度就要平均了 平均每步
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()# 分离梯度 不参与反向传播 计算有未来时间的误差范数
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)# 预测的不确定度和 [有效N数, F]
        loss = reg_loss + cls_loss
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss
    ### added by long
    def predition_unit_batch(self, data, batch_idx):
        y_hat0, pi = self(data)
        print("predition_unit_batch is called", flush=True)
        # print('unit print ',y_hat0.size())
        # y_hat = y_hat.unsqueeze(0)
        # pi = pi.unsqueeze(0)
        # pi = F.softmax(pi)
        y_hat0 = y_hat0.permute(1, 0, 2, 3)# 从模态，agent，轨迹点，坐标-->agent，模态，轨迹点，坐标 6 30 2
        # print('unit print ',y_hat0.size())
        y_hat =  y_hat0[:, :, :, :2]
        # y_hat_agent = y_hat[data['agent_index'], :, :, :2]
        # print('unit print y_hat ',y_hat.size())
        # pi_agent = pi[data['agent_index'], :]
        if self.rotate:
            data_angles = data['theta']# AV 的角度
            data_origin = data['origin']
            data_rotate_angle = data['rotate_angles']# 各自headings
            data_local_origin = data.positions[:, 19, :]#预测目标真值轨迹起点
            rotate_mat = torch.empty(data['num_nodes'], 2, 2, device=self.device)
            # rotate_mat = torch.empty(1, 2, 2, device=self.device)
            sin_vals = torch.sin(-data_angles)
            cos_vals = torch.cos(-data_angles)#负的角度
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals#原来的旋转矩阵 R^E_AV
            rotate_local = torch.empty(data['num_nodes'], 2, 2, device=self.device)
            # rotate_local = torch.empty(1, 2, 2, device=self.device)
            sin_vals_angle = torch.sin(-data_rotate_angle)
            cos_vals_angle = torch.cos(-data_rotate_angle)
            rotate_local[:, 0, 0] = cos_vals_angle
            rotate_local[:, 0, 1] = -sin_vals_angle
            rotate_local[:, 1, 0] = sin_vals_angle
            rotate_local[:, 1, 1] = cos_vals_angle  #agent车的姿态矩阵R^AV_AG的 逆矩阵R^AG_AV

            position_av = data['positions'] 


            for i in range(data['num_nodes']):#agent/actor，模态，轨迹点，坐标
                stacked_rotate_mat = torch.stack([rotate_mat[i]] *y_hat.shape[1], dim=0)
                stacked_rotate_local = torch.stack([rotate_local[i]] *y_hat.shape[1] , dim=0)

                # print("input shape:")
                # print(y_hat[i, :, :, :].shape)
                # print(stacked_rotate_mat.shape)
                # print(data_origin[i].shape)
                y_hat[i, :, :, :] =  torch.matmul( y_hat[i, :, :, :], stacked_rotate_local) + data_local_origin[i].unsqueeze(0).unsqueeze(0)
                
                y_hat[i, :, :, :] =  torch.matmul( y_hat[i, :, :, :], stacked_rotate_mat) + data_origin.unsqueeze(0).unsqueeze(0)
                position_av[i,:,:] = torch.matmul(position_av[i,:,:],rotate_mat[i])+ data_origin.unsqueeze(0).unsqueeze(0)
        return  y_hat, pi, data['seq_id'],position_av # 转到boot坐标下，pos_av是原始路径在boot下
    def predition_step(self, data, batch_idx):
        y_hat, pi = self(data)
        pi = F.softmax(pi,dim = 1)
        y_hat = y_hat.permute(1, 0, 2, 3)
        y_hat_agent = y_hat[data['agent_index'], :, :, :2]
        pi_agent = pi[data['agent_index'], :]
        if self.rotate:
            data_angles = data['theta']
            data_origin = data['origin']
            data_rotate_angle = data['rotate_angles'][data['agent_index']]
            data_local_origin = data.positions[data['agent_index'], 19, :]
            rotate_mat = torch.empty(data['agent_index'].shape[0], 2, 2, device=self.device)
            # rotate_mat = torch.empty(1, 2, 2, device=self.device)
            sin_vals = torch.sin(-data_angles)
            cos_vals = torch.cos(-data_angles)
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            rotate_local = torch.empty(data['agent_index'].shape[0], 2, 2, device=self.device)
            # rotate_local = torch.empty(1, 2, 2, device=self.device)
            sin_vals_angle = torch.sin(-data_rotate_angle)
            cos_vals_angle = torch.cos(-data_rotate_angle)
            rotate_local[:, 0, 0] = cos_vals_angle
            rotate_local[:, 0, 1] = -sin_vals_angle
            rotate_local[:, 1, 0] = sin_vals_angle
            rotate_local[:, 1, 1] = cos_vals_angle
            for i in range(data['agent_index'].shape[0]):
                stacked_rotate_mat = torch.stack([rotate_mat[i]] * self.num_modes, dim=0)
                stacked_rotate_local = torch.stack([rotate_local[i]] * self.num_modes, dim=0)
                # print("input shape:", y_hat_agent[i, :, :, :].shape, stacked_rotate_mat.shape, data_origin[i].shape)
                y_hat_agent[i, :, :, :] = torch.bmm(y_hat_agent[i, :, :, :], stacked_rotate_local) \
                                          + data_local_origin[i].unsqueeze(0).unsqueeze(0)
                y_hat_agent[i, :, :, :] = torch.bmm(y_hat_agent[i, :, :, :], stacked_rotate_mat) \
                                          + data_origin[i].unsqueeze(0).unsqueeze(0)
        return y_hat_agent, pi_agent, data['seq_id']
    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data) # 从模态，agent，轨迹点，坐标+不确定性 [F N1+N2+...N20 30 4] 整个batch
        # print('y_hat.shape',y_hat.shape)

        reg_mask = ~data['padding_mask'][:, self.historical_steps:]# reg true 的 才计算 也就是 padding false 的才计算 就是轨迹是有效的地方才计算 padd的位置没有真值
        # [F N 30 :2] [N, 30, 2] --> [F N 30 2] [1, N, 30, 2] = [F N 30]  对时间维度 30 进行求和 [F N] 存放不同agent 不同模态的 l2 norm
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N] 模态和node数
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]#best_mode挑选对应模态index，num_node 遍历所有node index
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])# 有效部分的最小的  NLL laplace loss 含有概率的  是多个障碍物的平均 各自一个模态轨迹 
        # 将验证损失 val_reg_loss 记录并显示在训练进度条和日志中。
        # 确保该指标在每个 epoch 结束时（而不是每个 batch 结束时）进行记录。
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)#展示每个样本的指标值

        agent_av_idx =  data['agent_index']
        # print('agent_av_idx.shape',agent_av_idx.shape)
        y_hat_agent = y_hat[:, agent_av_idx, :, : 2]
        y_agent = data.y[agent_av_idx]
        # 专门计算指定agent的fde
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)# [F,30,2]
        best_mode_agent = fde_agent.argmin(dim=0)
        # print('y_hat_agent.shape  ',y_hat_agent.shape) #torch.Size([6, 20, 30, 2])
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]#后面这个维度没什么必要
        # print('y_hat_best_agent.shape',y_hat_best_agent.shape) #torch.Size([20, 30, 2]) 
        # print('data.num_graphs in validation_step is ',data.num_graphs)# 20
        # 每个障碍物的best轨迹
        self.minADE.update(y_hat_best_agent, y_agent)# 20(batch_size)个
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        # validation_step --> update  validation_epoch_end --> compute
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))# 批次数平均
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
