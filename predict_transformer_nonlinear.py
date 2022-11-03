import os
from utils import get_freest_gpu
FREEST_GPU = get_freest_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FREEST_GPU)
from new_block_env import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import itertools
from torch.utils.data import random_split, DataLoader
import numpy as np
import sys
from tqdm import tqdm
import wandb
import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
import new_block_env_new
import ray
import psutil

NUM_TRAJ_PER_REWARD = 100
NUM_AGENTS_PER_GPU = 8
rng = np.random.default_rng()

def parse_args():
    parser = argparse.ArgumentParser(description='Train supervised IRL models')
    # parser.add_argument('--num-examples', type=int, default=4600,
    #                     help='Number of training examples')
    # parser.add_argument('--network-type', type=str, default='transformer',
    #                     help='Type of network to train')
    parser.add_argument('--save-model', default=True, action=argparse.BooleanOptionalAction,
                        help='Whether to save the model after every epoch')
    parser.add_argument('--save-to', '-st', type=str, default='nonlinear_model.parameters',
                        help='Destination for saved model')
    parser.add_argument('--num-epochs', '-ne', type=int, default=50)
    parser.add_argument('--batch-size', '-bs', type=int, default=8)
    parser.add_argument('--mlp', '-m', action='store_true')
    parser.add_argument('--evaluate', '-e', action='store_true', help='run for one epoch without updating')
    parser.add_argument('--saved-model', '-sm', type=str, default=None, help='location of saved model to load')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--wandb-project', '-wp', type=str, default='sirl')
    parser.add_argument('--permute-types', '-pt', action='store_true')
    parser.add_argument('--trajectory-sigmoid', '-nts', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--env', default="rings", type=str)
    parser.add_argument('--num-trajectory-layers', '-ntl', default=1, type=int)
    parser.add_argument('--num-state-layers', '-nsl', default=1, type=int)
    parser.add_argument('--state-hidden-size', '-shs', default=64, type=int)
    parser.add_argument('--trajectory-hidden-size', '-ths', default=2048, type=int)
    parser.add_argument('--lstm', '-l', action='store_true')
    parser.add_argument('--repeat-trajectory-calculations', '-rtc', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--trajectory-rep-dim', '-trd', default=3, type=int)
    parser.add_argument('--state-rep-dim', '-srd', default=3, type=int)
    parser.add_argument('--ground-truth-weights', '-gtw', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-shuffled', '-us', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--ground-truth-phi', '-gtp', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--rl-evaluation', '-re', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max-threads', '-mt', default=50, type=int, help="For RL evaluation only")
    parser.add_argument('--num-rings', '-nr', default=5, type=int, help="For ring env only")
    # parser.add_argument('--reward-batches', '-rb', default=False, action=argparse.BooleanOptionalAction)
    # XXX what is reward batches for?
    parser.add_argument('--data-prefix', '-dp', type=str, default="rings_multimove")
    parser.add_argument('--iid-traj', '-it', default=False, action=store_true, help="Don't do batching by task")
    parser.add_argument('--n', '-n', default=1, type=int, help="How many trajectories to train on (if iid traj, must be 1)")
    parser.add_argument('--rand-n', '-rn', default=False, action=argparse.BooleanOptionalAction, help="Train on a random number of trajectories within each task and epoch")
    parser.add_argument('--num-states', '-ns', type=int, help="Number of iid states to predict rewards of per task (defaults to all)")
    args = parser.parse_args()
    args.num_epochs = args.num_epochs if not args.evaluate else 1
    
    
    assert args.env == "rings", "remember to change the dataset names"
    assert not args.permute_types, "we're not sure this is correct, remember?"
    return args

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=151):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                  math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                  math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x

# class OtherPositionalEncoder(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(OtherPositionalEncoder, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [sequence length, batch size, embed dim]
#             output: [sequence length, batch size, embed dim]
#         Examples:
#             >>> output = pos_encoder(x)
#         """

#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)        


# Try replacing with LSTM
# Check trajectories (certainly don't have enough cap. to memorize illogical differences in dumb Hs)
# To that end, should also run on space invaders
class TrajectoryNetTransformer(nn.Module):
    def __init__(self, obs_size, horizon, num_transformer_layers, trajectory_hidden_size, trajectory_rep_dim, sigmoid=True):
        super().__init__()
        self.positional_encoder = PositionalEncoder(obs_size, max_seq_len=horizon+1) # not sure why this is +1 but we'll go with what we had
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=obs_size, nhead=10, dim_feedforward=trajectory_hidden_size, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_transformer_layers)
        self.linear1 = nn.Linear(obs_size, 50)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(50, trajectory_rep_dim)
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()
        self.obs_size = obs_size
        self.horizon = horizon

    def forward(self, states):
        x = self.positional_encoder(states)
        x = self.transformer(x)
        assert(x.shape[1] == self.horizon)
        assert(x.shape[2] == self.obs_size)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        #x = x.view(-1, 3)
        x = self.sigmoid(x)
        return x
    
class TrajectoryNetLSTM(nn.Module):
    def __init__(self, obs_size, num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(25*obs_size, 2048, num_lstm_layers, batch_first=True)
        self.linear1 = nn.Linear(2048, 16)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(16, 3)
        
    def forward(self, states):
        x, _ = self.lstm(states)
        x = x[:, -1, :]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
class RewardNet(nn.Module):
    def __init__(self, trajectory_rep_dim, state_rep_dim, hidden_size):
        super().__init__()
        combined_dim = trajectory_rep_dim + state_rep_dim
        self.linear1 = nn.Linear(combined_dim, hidden_size)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, trajectory_rep, state_rep):
        combined_rep = torch.cat((trajectory_rep, state_rep), axis=-1)
        x = self.linear1(combined_rep)
        x = self.relu(x)
        x = self.linear2(x)
        return x.squeeze()
    
class StateNet(nn.Module):
    def __init__(self, state_rep_dim, hidden_size, num_layers, obs_size):
        super().__init__()
        modules = []
        modules.append(nn.Linear(obs_size, hidden_size))
        for i in range(num_layers-1):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(hidden_size, state_rep_dim))
        self.net = nn.Sequential(*modules)
    
    def forward(self, state):
        return self.net(state)
    
class NonLinearNet(nn.Module):
    def __init__(self, trajectory_rep_dim, 
                 state_rep_dim, 
                 state_hidden_size, 
                 reward_hidden_size, 
                 trajectory_hidden_size,
                 obs_size, 
                 horizon,
                 num_trajectory_layers, 
                 num_state_layers, 
                 mlp=False, 
                 trajectory_sigmoid=False, 
                 lstm=False, 
                 repeat_trajectory_calculations=False,
                 ground_truth_phi=False):
        
        super().__init__()
        assert trajectory_rep_dim == state_rep_dim, "Non-matching rep dims not yet implemented!"
        assert not repeat_trajectory_calculations, "Really shouldn't be repeating calculations"
        if lstm:
            self.trajectory_encoder = TrajectoryNetLSTM(obs_size, num_trajectory_layers)
        else:
            self.trajectory_encoder = TrajectoryNetTransformer(obs_size, horizon, num_trajectory_layers, trajectory_hidden_size, trajectory_rep_dim, trajectory_sigmoid)
        # self.trajectory_encoder = SITrajectoryNet()
        self.state_encoder = StateNet(state_rep_dim, state_hidden_size, num_state_layers, obs_size)
        self.mlp = mlp
        self.horizon = horizon
        if repeat_trajectory_calculations:
            self.forward = self.forward_repeat
        else:
            self.forward = self.forward_no_repeat
        if mlp:
            self.reward_layer = RewardNet(trajectory_rep_dim, state_rep_dim, reward_hidden_size)
        else:
            assert(trajectory_rep_dim == state_rep_dim)
        self.ground_truth_phi = ground_truth_phi
        if ground_truth_phi:
            self.inv_distances = torch.zeros((5, 5)).float().cuda()
            for x, y in itertools.product(range(5), repeat=2):
                if x == 2 and y == 2:
                    continue
                self.inv_distances[x, y] = 1/(np.abs(x-2) + np.abs(y-2))
            self.inv_distances[2, 2] = 20.
            self.inv_distances = torch.flatten(self.inv_distances)

    def forward_repeat(self, trajectory, state):
        assert False, "no"
        trajectory_rep = self.trajectory_encoder(trajectory)
        state_rep = self.state_encoder(state)
        if self.mlp:
            reward = self.reward_layer(trajectory_rep, state_rep)
        else:
            reward = torch.einsum('bs,bs->b', trajectory_rep, state_rep)
        return reward
    
    def forward_no_repeat(self, trajectories, weights=None):
        # trajectories: (bsize, L, |S|)
        if weights is None:
            trajectory_rep = self.trajectory_encoder(trajectories) # (bsize, rep_dim)
        else:
            trajectory_rep = weights
        states = trajectories.view(-1, trajectories.shape[-1]) # (bsize*L, |S|)
        if not self.ground_truth_phi:
            state_rep = self.state_encoder(states) # (bsize*L, rep_dim)
        else:
            states_unflattened = states.view(-1, 25, 4)
            inv_distances_unsqueezed = self.inv_distances.unsqueeze(0).unsqueeze(-1)
            inv_distances_expanded = inv_distances_unsqueezed.expand(states.shape[0], -1, 4)
            state_rep_with_empties = torch.einsum('bst, bst -> bt', states_unflattened, inv_distances_expanded) # Multiply everything then sum across the 25 squares (s)
            state_rep = state_rep_with_empties[:, :3]
        trajectory_rep_expanded = trajectory_rep.unsqueeze(1).expand(-1, self.horizon, -1) # (bsize, L, rep_dim)
        trajectory_rep_flattened = trajectory_rep_expanded.reshape(-1, trajectory_rep_expanded.shape[-1]) # (bsize*L, rep_dim)
        if self.mlp:
            reward = self.reward_layer(trajectory_rep_flattened, state_rep)
        else:
            reward = torch.einsum('bs,bs->b', trajectory_rep_flattened, state_rep)
        return reward.view(trajectories.shape[0], -1)
    
class NonlinearDataset(torch.utils.data.Dataset):
    def __init__(self, states, rewards, weights, num_classes=4):
        self.rewards = torch.Tensor(rewards).to(torch.float) #(n_examples, L)
        if num_classes == 0: # Indicates continuous environment, i.e. rings (at the moment)
            self.states = torch.Tensor(states).to(torch.float) #(n_examples, L, S);
        else: # Discrete environment
            self.states = torch.Tensor(states).to(torch.int64) #(n_examples, L, S); might need to be float
            self.states = F.one_hot(self.states, num_classes=num_classes).view(-1, 150, 25*num_classes)
        self.weights = weights # usually None
        if weights is not None:
            self.weights = torch.Tensor(weights).to(torch.float)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        if self.weights is not None:
            return self.states[index].cuda().to(torch.float), self.rewards[index].cuda(), self.weights[index].cuda() # returns a whole (L, S), (L, 1) trajectory
        else:
            return self.states[index].cuda().to(torch.float), self.rewards[index].cuda(), 0 # 0 here is in effect None, just not allowed to use None

# Dataset for data organized by task (i.e. reward)
# Allows for us to do n-shot, as well as using iid states separate from the
# demonstration trajectory for inference
# Currently only implemented for rings
class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, states, rewards, weights, n=1, rand_n=False, num_states=None):
        self.rewards = torch.Tensor(rewards).to(torch.float) #(n_tasks, n_examples, L)
        self.states = torch.Tensor(states).to(torch.float) #(n_tasks, n_examples, L, S)
        self.weights = weights # usually None
        self.n = n
        self.rand_n = rand_n # if true, uses self.n as max n
        if num_states is None:
            self.num_states = self.states.shape[1]*self.states.shape[2] # e.g. 100*T
        else:
            self.num_states = num_states
        if weights is not None:
            self.weights = torch.Tensor(weights).to(torch.float)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        trajs = self.states[index]
        rewards = self.rewards[index]
        if self.rand_n:
            n = rng.integers(self.n)
        else:
            n = self.n
        shuffled_trajs = rng.permutation(trajs)
        return_trajs = shuffled_trajs[:n]
        all_states = np.reshape(trajs, (-1, trajs.shape[-1]))
        all_rewards = np.reshape(rewards, (-1,))
        shuffled_idxs = rng.permutation(np.arange(len(all_states)))
        shuffled_states = all_states[shuffled_idxs]
        shuffled_rewards = rewards[shuffled_idxs]
        return_states = shuffled_states[:self.num_states]
        return_rewards = shuffled_rewards[:self.num_states]
        if self.weights is not None:
            return return_states.cuda(), return_rewards.cuda(), self.weights[index].cuda()
        else:
            return return_states.cuda(), return_rewards.cuda(), 0 # 0 here is in effect None, just not allowed to use None
    
def get_splits(args):
    prefix = "data/"
    prefix += "shuffled_" if args.use_shuffled else ""
    state_path = prefix + args.data_prefix + "_states.npy"
    reward_path = prefix + args.data_prefix + "_rewards.npy"
    # if args.space_invaders:
    #     state_file = os.path.join('spaceinvaders', f"states_{args.num_examples}.npy")
    #     reward_file = os.path.join('spaceinvaders', f"rewards_{args.num_examples}.npy")
    #     assert not args.ground_truth_weights, "not implemented yet"
    # else:
    #     state_file = prefix + f"states_{args.num_examples}.npy"
    #     reward_file = prefix + f"rewards_{args.num_examples}.npy"
    states = np.load(state_path, allow_pickle=True)
    rewards = np.load(reward_path, allow_pickle=True)
    if len(states.shape) == 3:
        # Data is long-form, i.e. not already grouped by task; group by task first
        assert args.num_examples % NUM_TRAJ_PER_REWARD == 0
        num_rewards = args.num_examples // NUM_TRAJ_PER_REWARD
        states_by_reward = np.reshape(states, (num_rewards, -1, states.shape[-2], states.shape[-1]))
        rewards_by_reward = np.reshape(rewards, (num_rewards, -1, rewards.shape[-1]))    
    else:
        states_by_reward = states
        rewards_by_reward = rewards
        num_rewards = len(states_by_reward)
    
    # XXX pretty sure this isn't correct right now
    # if args.permute_types:
    #     for i in range(num_rewards):
    #         permutation = np.arange(3, 6) if args.space_invaders else np.arange(1, 4)
    #         rng.shuffle(permutation)
    #         base_classes = np.arange(3 if args.space_invaders else 1)
    #         assert not args.space_invaders, "I think the next line is wrong order here"
    #         permutations_plus = np.concatenate((permutation, base_classes))
    #         states_by_reward[i] = permutations_plus[states_by_reward[i]]
    
    # Create train/val split by creating list of task indices, shuffling, and then indexing
    reward_idx_list = np.arange(num_rewards)
    rng.shuffle(reward_idx_list)
    train_length = int(num_rewards*0.8)
    val_length = num_rewards - train_length
    train_idxs = reward_idx_list[:train_length]
    val_idxs = reward_idx_list[train_length:]
    train_states = states_by_reward[train_idxs]
    train_rewards = rewards_by_reward[train_idxs]
    val_states = states_by_reward[val_idxs]
    val_rewards = rewards_by_reward[val_idxs]
    if not args.iid_traj:
        train_dataset = TaskDataset(
    
    if args.ground_truth_weights or args.rl_evaluation:
        weights_file = prefix + f"weights_{args.num_examples}.npy"
        weights = np.load(weights_file, allow_pickle=True)
        weights_by_reward = np.reshape(weights, (num_rewards, -1, weights.shape[-1]))
        train_weights = weights_by_reward[train_idxs]
        train_weights_flattened = np.reshape(train_weights, (-1, train_weights.shape[-1]))
        val_weights = weights_by_reward[val_idxs]
        val_weights_flattened = np.reshape(val_weights, (-1, val_weights.shape[-1]))
    else:
        train_weights_flattened, val_weights_flattened = None, None
    
    # TODO non iid_traj one, and add non-flattened weights        
            
    if args.iid_traj:
        # Flatten for final dataloader (can change this eventually if we want to train in pairs or more)
        train_states_flattened = np.reshape(train_states, (-1, train_states.shape[-2], train_states.shape[-1]))
        train_rewards_flattened = np.reshape(train_rewards, (-1, train_rewards.shape[-1]))
        val_states_flattened = np.reshape(val_states, (-1, val_states.shape[-2], val_states.shape[-1]))
        val_rewards_flattened = np.reshape(val_rewards, (-1, val_rewards.shape[-1]))
            
        train_dataset = NonlinearDataset(train_states_flattened, train_rewards_flattened, train_weights_flattened, num_classes)
        val_dataset = NonlinearDataset(val_states_flattened, val_rewards_flattened, val_weights_flattened, num_classes)
    
    # training_data, validation_data = random_split(dataset, [train_length, val_length])
    # print("Num training", len(training_data))
    # print("Num validation", len(validation_data))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, validation_dataloader

@ray.remote(num_gpus=1/NUM_AGENTS_PER_GPU)
def evaluate_rl(traj_rep, state_encoder, weights):
    means = np.zeros((10, 3), dtype=float)
    for i in range(10):
        pred_env = new_block_env_new.NewBlockEnv(traj_rep, state_encoder)
        ground_truth_env = new_block_env_new.NewBlockEnv(weights)
        pred_model = stable_baselines3.PPO("MlpPolicy", pred_env, verbose=1)
        pred_model.learn(total_timesteps=100000)
        ground_truth_model = stable_baselines3.PPO("MlpPolicy", ground_truth_env, verbose=1)
        ground_truth_model.learn(total_timesteps=100000)
        random_model = stable_baselines3.PPO("MlpPolicy", ground_truth_env, verbose=1)
        mean_ground_truth_pred_model_ret = evaluate_policy(pred_model, ground_truth_env, n_eval_episodes=100)
        mean_ground_truth_ret = evaluate_policy(ground_truth_model, ground_truth_env, n_eval_episodes=100)
        mean_random_ret = evaluate_policy(random_model, ground_truth_env, n_eval_episodes=100)
        means[i] = [mean_ground_truth_pred_model_ret[0], mean_ground_truth_ret[0], mean_random_ret[0]]
    return np.mean(means, axis=0)

def train(args):
    wandb.init(project='sirl')
    wandb.config.update(args)
    
    train_dataloader, validation_dataloader = get_splits(args)
    
    # torch.cuda.set_device(f'cuda:{FREEST_GPU}')

    if "grid" in args.env:
        obs_size = 4*25
        horizon = 150
    elif "space" in args.env:
        obs_size = 6*25
        horizon = 150
    elif "ring" in args.env or "object" in env:
        obs_size = 2*args.num_rings
        horizon = 50
    else:
        raise NotImplementedError

    net = NonLinearNet(args.trajectory_rep_dim, 
                       args.state_rep_dim, 
                       args.state_hidden_size, 
                       64, 
                       args.trajectory_hidden_size, 
                       obs_size, 
                       horizon,
                       args.num_trajectory_layers, 
                       args.num_state_layers, 
                       mlp=args.mlp, 
                       trajectory_sigmoid=args.trajectory_sigmoid, 
                       lstm=args.lstm, 
                       repeat_trajectory_calculations=args.repeat_trajectory_calculations, 
                       ground_truth_phi = args.ground_truth_phi).cuda()
    if args.saved_model:
        net.load_state_dict(torch.load(args.saved_model))

    optimizer = torch.optim.Adam(net.parameters()) if not args.evaluate else None
    loss_func = nn.MSELoss()
    
    # assert args.rl_evaluation == args.ground_truth_weights, "You're using ground truth weights, are you sure you want to do that?"
    
    # TQDM inspiration from https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e

    for epoch in range(args.num_epochs):
        avg_loss = 0
        n_samples = 0
        if args.evaluate:
            net.eval()
        else:
            net.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for states_batch, rewards_batch, weights_batch in tepoch:
                # if torch.any(torch.all(weights_batch < 0, axis=1)):
                    # breakpoint()
                # states_batch: (bsize, horizon, obs_size)
                # rewards_batch: (bsize, horizon)
                tepoch.set_description(f"Epoch {epoch}")
                # In effect, we now batch by batch *and* t
                if not args.evaluate:
                    optimizer.zero_grad()
                if not args.repeat_trajectory_calculations:
                    if args.ground_truth_weights:
                        prediction = net.forward(states_batch, weights_batch)
                    else:
                        prediction = net.forward(states_batch)
                else:
                    trajectory_batch = states_batch.unsqueeze(1).expand(-1, horizon, horizon, obs_size) #repeat trajectory horizon times
                    trajectory_batch = trajectory_batch.reshape(-1, horizon, obs_size) # return to batch form: (bsize*L, L, obs_size)
                    states_batch = states_batch.view(-1, obs_size) # (bsize*L, obs_size)
                    rewards_batch = rewards_batch.view(-1) # bsize * L
                    prediction = net.forward(trajectory_batch, states_batch)
                loss = loss_func(prediction, rewards_batch)
                wandb.log({'train loss': loss})
                # loss.backward()
                if args.verbose and tepoch.n % 100 == 0:
                    print("Predicted vs actual rewards: \n", np.array(list(zip(prediction.cpu().detach().numpy(), rewards_batch.cpu().detach().numpy()))))
                    print("Maximum error: \n", torch.max((prediction-rewards_batch)**2))
                    worst = torch.argmax((prediction-rewards_batch)**2)
                    print("Argmax of error: \n", prediction.flatten()[worst], rewards_batch.flatten()[worst])
                    print("Trajectory reps: \n", net.trajectory_encoder(states_batch))
                    # for name, param in net.named_parameters():
                    #     if param.grad is not None:
                    #         grad = torch.max(param.grad)
                    #     else:
                    #         grad = None
                    #     print(name, param, grad)
                avg_loss += loss.item() * args.batch_size * horizon
                n_samples += args.batch_size * horizon
                if not args.evaluate:
                    loss.backward()
                    optimizer.step()
                if args.rl_evaluation and (tepoch.n % 100 == 0 or tepoch.n % 101 == 0):
                    # n_threads = min(args.max_threads, psutil.cpu_count()-1)
                    traj_reps = net.trajectory_encoder(states_batch).cpu().detach().numpy()
                    ray.init(num_cpus=NUM_AGENTS_PER_GPU, num_gpus=1)
                    rets = ray.get([evaluate_rl.remote(traj_rep, net.state_encoder, weights) for traj_rep, weights in zip(traj_reps, weights_batch.detach().cpu().numpy())])
                    for ret in rets:
                        wandb.log({"rl_pred": ret[0], "rl_true": ret[1], "rl_random": ret[2]})
                    ray.shutdown()
                        
                tepoch.set_postfix({"avg loss": avg_loss/n_samples})

        avg_loss /= len(train_dataloader)*horizon*args.batch_size
        #n_correct /= (3 * len(training))

        # UNCOMMENT TO SAVE MODEL
        if args.save_model and not args.evaluate:
            torch.save(net.state_dict(), args.save_to)

        avg_training_loss = avg_loss
        print(f"Average training loss: {avg_training_loss}")
        wandb.log({'average train loss': avg_training_loss})
        avg_loss = 0
        n_samples = 0
        net.eval()
        with torch.no_grad():
            with tqdm(validation_dataloader, unit="batch") as tepoch:
                for states_batch, rewards_batch, weights_batch in tepoch:
                    # states_batch: (bsize, horizon, 100)
                    # rewards_batch: (bsize, horizon)
                    tepoch.set_description("Epoch {epoch} (validation)")
                    # In effect, we now batch by batch *and* t
                    if not args.repeat_trajectory_calculations:
                        if args.ground_truth_weights:
                            prediction = net.forward(states_batch, weights_batch)
                        else:
                            prediction = net.forward(states_batch)
                    else:
                        trajectory_batch = states_batch.unsqueeze(1).expand(-1, horizon, horizon, 25*num_classes) #repeat trajectory horizon times
                        trajectory_batch = trajectory_batch.reshape(-1, horizon, 25*num_classes) # return to batch form: (bsize*L, L, 100)
                        states_batch = states_batch.view(-1, 25*num_classes) # (bsize*L, 100)
                        rewards_batch = rewards_batch.view(-1) # bsize * L
                        prediction = net.forward(trajectory_batch, states_batch)
                    loss = loss_func(prediction, rewards_batch)
                    avg_loss += loss.item() * args.batch_size * horizon
                    n_samples += args.batch_size*horizon
                    tepoch.set_postfix({"avg validation loss": avg_loss/n_samples})
        avg_loss /= len(validation_dataloader)*horizon*args.batch_size
        wandb.log({'average validation loss': avg_loss})
    if not args.evaluate:
        torch.save(net.state_dict(), args.save_to)

if __name__ == "__main__":
    args = parse_args()
    train(args)