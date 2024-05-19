"""
DO NOT MODIFY BESIDES HYPERPARAMETERS 
"""
import torch
import numpy as np

from planning_mbrl import collect_traj_MBRL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(env, model, plan_mode, num_validation_runs=10, episode_length=200, render=False, mpc_horizon=None, n_samples_mpc=None, reward_fn=None):
    success_count = 0
    rewards_suc = 0
    rewards_all = 0
    for k in range(num_validation_runs):
        o = env.reset()
        path = collect_traj_MBRL(
            env,
            model,
            plan_mode,
            episode_length=episode_length,
            render=render,
            mpc_horizon=mpc_horizon,
            n_samples_mpc=n_samples_mpc,
            device=device,
            reward_fn=reward_fn
        )
        success = np.linalg.norm(env.get_body_com("fingertip") - env.get_body_com("target"))<0.1

        if success:
            success_count += 1
            rewards_suc += np.sum(path['rewards'])
        rewards_all += np.sum(path['rewards'])
        print(f"test {k}, success {success}, reward {np.sum(path['rewards'])}")
    print("Success rate: ", success_count/num_validation_runs)
    print("Average reward (success only): ", rewards_suc/max(success_count, 1))
    print("Average reward (all): ", rewards_all/num_validation_runs)