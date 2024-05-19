import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import argparse
import random
from utils import DeterministicDynamicsModel, set_random_seed
from planning_mbrl import simulate_mbrl
from evaluate import evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='single', help='choose type of model')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--plan_mode', type=str, default='random_mpc', help='choose planning method')
    parser.add_argument('--render',  action='store_true', default=False)

    args = parser.parse_args()
    
    if args.render:
        import os
        os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGLEW.so"
    
    # Environment and reward definition
    env = gym.make("Reacher-v2")
    def reward_fn_reacher(state, action):
        cos_theta = state[:, :2]
        sin_theta = state[:, 2:4]
        qpos = state[:, 4:6]
        qvel = state[:, 6:8]
        vec = state[:, 8:11]

        reward_dist = -torch.norm(vec, dim=1)
        reward_ctrl = -torch.square(action).sum(dim=1)

        reward = reward_dist + reward_ctrl
        reward = reward[:, None]
        return reward
    reward_fn = reward_fn_reacher
    
    # Define dynamics model
    hidden_dim_model = 64
    hidden_depth_model = 2
    if args.model_type == 'single':
        model = DeterministicDynamicsModel(env.observation_space.shape[0] + env.action_space.shape[0], env.observation_space.shape[0], hidden_dim=hidden_dim_model, hidden_depth=hidden_depth_model)
        model.to(device)
    elif args.model_type == 'ensemble':
        num_ensembles = 5
        model = []
        for model_id in range(num_ensembles):
            curr_model = DeterministicDynamicsModel(env.observation_space.shape[0] + env.action_space.shape[0], env.observation_space.shape[0], hidden_dim=hidden_dim_model, hidden_depth=hidden_depth_model)
            curr_model.to(device)
            model.append(curr_model)
    else:
        raise NotImplementedError("No other model types implemented")
    
    # Training hyperparameters
    num_epochs=15
    max_path_length=50
    batch_size=250 #5000
    num_agent_train_epochs_per_iter=10 #100
    num_traj_per_iter = batch_size // max_path_length
    gamma=0.99
    print_freq=1
    capacity=100000
    mpc_horizon = 10
    n_samples_mpc = 1000

    if not args.test:
        # Training and model saving code
        simulate_mbrl(env, model, plan_mode=args.plan_mode, num_epochs=num_epochs, max_path_length=max_path_length, mpc_horizon=mpc_horizon,
                    n_samples_mpc=n_samples_mpc, batch_size=batch_size, num_agent_train_epochs_per_iter=num_agent_train_epochs_per_iter, capacity=capacity, num_traj_per_iter=num_traj_per_iter, gamma=gamma, print_freq=print_freq, device = "cuda", reward_fn=reward_fn)
        if type(model) is list:
            for model_idx, curr_model in enumerate(model):
                torch.save(curr_model.state_dict(), f'{args.model_type}_{args.plan_mode}_{model_idx}.pth')
        else:
            torch.save(model.state_dict(), f'{args.model_type}_{args.plan_mode}.pth')
    else:
        print('loading pretrained mbrl')
        if type(model) is list:
            for model_idx in range(len(model)):

                model[model_idx].load_state_dict(torch.load(f'{args.model_type}_{args.plan_mode}_{model_idx}.pth'))
        else:
            model.load_state_dict(torch.load(f'{args.model_type}_{args.plan_mode}.pth'))

    evaluate(env, model, plan_mode=args.plan_mode, mpc_horizon=mpc_horizon, n_samples_mpc=n_samples_mpc, num_validation_runs=100, episode_length=max_path_length, render=args.render, reward_fn=reward_fn)