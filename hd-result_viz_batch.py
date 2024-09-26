import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import  List
from models.hivt import HiVT
import os
import random
import torch.nn.functional as F


def plot_single_vehicle(
    sample_past_trajectory: np.ndarray,  # (20, 2)
    sample_groundtruth: np.ndarray,  # (20, 2) 
    sample_forecasted_trajectories: List[np.ndarray],  # List[(30, 2)]
    ):

    ## Plot history
    plt.plot(
        sample_past_trajectory[:, 0],
        sample_past_trajectory[:, 1],
        color="#ECA154",
        label="Past Trajectory",
        alpha=1,
        linewidth=2.5,
        zorder=15,
        ls = '-'
    )

    ## Plot future
    plt.plot(
        sample_groundtruth[:, 0],
        sample_groundtruth[:, 1],
        color="#d33e4c",
        label="Ground Truth",
        alpha=1,
        linewidth=2.5,
        zorder=20,
        ls = "--"
    )

    ## Plot prediction
    for j in range(len(sample_forecasted_trajectories)):
        plt.plot(
            sample_forecasted_trajectories[j][:, 0],
            sample_forecasted_trajectories[j][:, 1],
            color="#007672",
            label="Forecasted Trajectory",
            alpha=1,
            linewidth=2.0,
            zorder=15,
            ls = "--"
        )
        
        # Plot the end marker for forcasted trajectories
        plt.arrow(
            sample_forecasted_trajectories[j][-2, 0], 
            sample_forecasted_trajectories[j][-2, 1],
            sample_forecasted_trajectories[j][-1, 0] - sample_forecasted_trajectories[j][-2, 0],
            sample_forecasted_trajectories[j][-1, 1] - sample_forecasted_trajectories[j][-2, 1],
            color="#007672",
            label="Forecasted Trajectory",
            alpha=1,
            linewidth=2.5,
            zorder=15,
            head_width=1.1,
        )
        
    
    # Plot the end marker for history
    plt.arrow(
            sample_past_trajectory[-2, 0], 
            sample_past_trajectory[-2, 1],
            sample_past_trajectory[-1, 0] - sample_past_trajectory[-2, 0],
            sample_past_trajectory[-1, 1] - sample_past_trajectory[-2, 1],
            color="#ECA154",
            label="Past Trajectory",
            alpha=1,
            linewidth=2.5,
            zorder=25,
            head_width=1.0,
        )

    ## Plot the end marker for future
    plt.arrow(
            sample_groundtruth[-2, 0], 
            sample_groundtruth[-2, 1],
            sample_groundtruth[-1, 0] - sample_groundtruth[-2, 0],
            sample_groundtruth[-1, 1] - sample_groundtruth[-2, 1],
            color="#d33e4c",
            label="Ground Truth",
            alpha=1,
            linewidth=2.5,
            zorder=25,
            head_width=1.0,
        )

model_unit = HiVT.load_from_checkpoint('/home/alon/Learning/HiVT/lightning_logs/version_57/checkpoints/epoch=34-step=269849.ckpt', parallel=True)

info_ = model_unit.eval()

folder_path = '/home/alon/Learning/HiVT/data_root/val/processed/'
files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
random.seed(2024)
random.seed(2025)
random.seed(2026)
random.shuffle(files)
cnt = 0
for file_name in files:

    if cnt > 10 :
        break
    file_path = os.path.join(folder_path, file_name)
    print(f"Loading file: {file_name}")
    test_data = torch.load(file_path)
    cnt = cnt + 1


    with torch.no_grad(): 
        y_hat_agent, pi_agent, seq_id ,position_sce= model_unit.predition_unit_batch(test_data,batch_idx=0)
    y_hat_agent = y_hat_agent.permute(0, 1, 2, 3)
    print('y_hat.size = ',y_hat_agent.size())
    pi_agent = F.softmax (pi_agent, dim=-1)
    pred_traj_np = y_hat_agent.cpu().numpy()
    full_traj = position_sce
    if full_traj.is_cuda:
        full_traj = full_traj.cpu()

    full_traj_np = full_traj.numpy()
    raw_lane_pos = test_data['raw_lane_pos']
    raw_lane_pos_np = raw_lane_pos.cpu().numpy()
    index_agent = test_data['agent_index']
    av_agent = test_data['av_index']
    complete_idx = test_data['complete_samples']
    
    for i in  complete_idx:
        print('pi_agent = ', pi_agent[i])
        actor_to_show = i
        sample_past_trajectory = full_traj[actor_to_show,:20,:]
        sample_groundtruth = full_traj[actor_to_show,20:,:]
        full_truth = full_traj_np[actor_to_show]
        sample_forecasted_trajectories = [pred_traj_np[actor_to_show][i] for i in range(3)]
        # orange red green: past future prediction
        plot_single_vehicle(sample_past_trajectory,sample_groundtruth,sample_forecasted_trajectories)
        plt.scatter(
        raw_lane_pos_np[:, 0],
        raw_lane_pos_np[:, 1],
        color="#000000",
        label="Lane",
        alpha=1,
        s = 2,
        zorder=10)
        # plt.axis('equal')
        # plt.show()
        # plt.close()各自一张图
        plt.title(f"{file_name}")
        plt.show()
        plt.close()
