from new_block_env import *
from utils import get_freest_gpu
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import random_split, DataLoader
import numpy as np
import sys
from tqdm import tqdm
import wandb

NUM_TRAJ_PER_REWARD = 10000
transformer_dimensions = 100
rng = np.random.default_rng()

def parse_args():
    parser = argparse.ArgumentParser(description='Train supervised IRL models')
    parser.add_argument('--num-examples', type=int, default=4600,
                        help='Number of training examples')
    # parser.add_argument('--network-type', type=str, default='transformer',
    #                     help='Type of network to train')
    parser.add_argument('--save-model', type=bool, default=True,
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
    # parser.add_argument('--dataset-file', '-df', type=str, default="examples_test_4600.npy")
    args = parser.parse_args()
    args.num_epochs = args.num_epochs if not args.evaluate else 1
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

class TrajectoryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_encoder = PositionalEncoder(transformer_dimensions)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dimensions, nhead=10, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.linear1 = nn.Linear(100, 50)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(50, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, states):
        x = self.positional_encoder(states)
        x = self.transformer(x)
        assert(x.shape[1] == 150)
        assert(x.shape[2] == 100)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        #x = x.view(-1, 3)
        x = self.sigmoid(x)
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
    def __init__(self, state_rep_dim, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(100, hidden_size)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, state_rep_dim)
    
    def forward(self, state):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
class NonLinearNet(nn.Module):
    def __init__(self, trajectory_rep_dim, state_rep_dim, state_hidden_size, reward_hidden_size, mlp=False):
        super().__init__()
        assert trajectory_rep_dim == state_rep_dim == 3, "Variable rep dims not yet implemented!"
        self.trajectory_encoder = TrajectoryNet()
        self.state_encoder = StateNet(state_rep_dim, state_hidden_size)
        self.mlp = mlp
        if mlp:
            self.reward_layer = RewardNet(trajectory_rep_dim, state_rep_dim, reward_hidden_size)
        else:
            assert(trajectory_rep_dim == state_rep_dim)

    def forward(self, trajectory, state):
        trajectory_rep = self.trajectory_encoder(trajectory)
        state_rep = self.state_encoder(state)
        if self.mlp:
            reward = self.reward_layer(trajectory_rep, state_rep)
        else:
            reward = torch.einsum('bs,bs->b', trajectory_rep, state_rep)
        # breakpoint()
        return reward
    
class NonlinearDataset(torch.utils.data.Dataset):
    def __init__(self, states, rewards):
        self.states = torch.Tensor(states).to(torch.int64) #(n_examples, L, S); might need to be float
        self.rewards = torch.Tensor(rewards).to(torch.float) #(n_examples, L)
        self.states = F.one_hot(self.states, num_classes=4).view(-1, 150, 100)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        return self.states[index].cuda().to(torch.float), self.rewards[index].cuda() # returns a whole (L, S), (L, 1) trajectory
    
def get_splits(args):
    state_file = f"states_{args.num_examples}.npy"
    reward_file = f"rewards_{args.num_examples}.npy"
    states = np.load(state_file, allow_pickle=True)
    rewards = np.load(reward_file, allow_pickle=True)
    assert args.num_examples % NUM_TRAJ_PER_REWARD == 0
    num_rewards = args.num_examples // NUM_TRAJ_PER_REWARD
    train_length = int(num_rewards*0.8)
    val_length = num_rewards - train_length
    reward_idx_list = np.arange(num_rewards)
    rng.shuffle(reward_idx_list)
    states_by_reward = np.reshape(states, (num_rewards, -1, states.shape[-2], states.shape[-1]))
    rewards_by_reward = np.reshape(rewards, (num_rewards, -1, rewards.shape[-1]))
    train_idxs = reward_idx_list[:train_length]
    val_idxs = reward_idx_list[train_length:]
    train_states = states_by_reward[train_idxs]
    train_rewards = rewards_by_reward[train_idxs]
    train_states_flattened = np.reshape(train_states, (-1, train_states.shape[-2], train_states.shape[-1]))
    train_rewards_flattened = np.reshape(train_rewards, (-1, train_rewards.shape[-1]))
    val_states = states_by_reward[val_idxs]
    val_rewards = rewards_by_reward[val_idxs]
    val_states_flattened = np.reshape(val_states, (-1, val_states.shape[-2], val_states.shape[-1]))
    val_rewards_flattened = np.reshape(val_rewards, (-1, val_rewards.shape[-1]))
    train_dataset = NonlinearDataset(train_states_flattened, train_rewards_flattened)
    val_dataset = NonlinearDataset(val_states_flattened, val_rewards_flattened)
    
    # training_data, validation_data = random_split(dataset, [train_length, val_length])
    # print("Num training", len(training_data))
    # print("Num validation", len(validation_data))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, validation_dataloader

def train(args):
    wandb.init(project='sirl')
    wandb.config.update(args)
    
    train_dataloader, validation_dataloader = get_splits(args)
    
    freest_gpu = get_freest_gpu()
    torch.cuda.set_device(f'cuda:{freest_gpu}')

    net = NonLinearNet(3, 3, 64, 64, mlp=args.mlp).cuda()
    if args.saved_model:
        net.load_state_dict(torch.load(args.saved_model))

    optimizer = torch.optim.Adam(net.parameters()) if not args.evaluate else None
    loss_func = nn.MSELoss()

    # TQDM inspiration from https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
    for epoch in range(args.num_epochs):

        avg_loss = 0
        n_samples = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for states_batch, rewards_batch in tepoch:
                # states_batch: (bsize, 150, 100)
                # rewards_batch: (bsize, 150)
                tepoch.set_description(f"Epoch {epoch}")
                # In effect, we now batch by batch *and* t
                trajectory_batch = states_batch.unsqueeze(1).expand(-1, 150, 150, 100) #repeat trajectory 150 times
                trajectory_batch = trajectory_batch.reshape(-1, 150, 100) # return to batch form: (bsize*L, L, 100)
                states_batch = states_batch.view(-1, 100) # (bsize*L, 100)
                rewards_batch = rewards_batch.view(-1) # bsize * L
                if not args.evaluate:
                    optimizer.zero_grad()
                prediction = net.forward(trajectory_batch, states_batch)
                loss = loss_func(prediction, rewards_batch)
                wandb.log({'train loss': loss})
                # loss.backward()
                if args.verbose and tepoch.n % 100 == 0:
                    print(prediction)
                    print(rewards_batch)
                    print(torch.max((prediction-rewards_batch)**2))
                    worst = torch.argmax((prediction-rewards_batch)**2)
                    print(prediction[worst], rewards_batch[worst])
                    # for name, param in net.named_parameters():
                    #     if param.grad is not None:
                    #         grad = torch.max(param.grad)
                    #     else:
                    #         grad = None
                    #     print(name, param, grad)
                avg_loss += loss.item() * args.batch_size * 150
                n_samples += args.batch_size * 150
                if not args.evaluate:
                    loss.backward()
                    optimizer.step()
                tepoch.set_postfix({"avg loss": avg_loss/n_samples})

        avg_loss /= len(train_dataloader)*150*args.batch_size
        #n_correct /= (3 * len(training))

        # UNCOMMENT TO SAVE MODEL
        if args.save_model and not args.evaluate:
            torch.save(net.state_dict(), args.save_to)

        avg_training_loss = avg_loss
        print(f"Average training loss: {avg_training_loss}")
        wandb.log({'average train loss': avg_training_loss})
        avg_loss = 0
        n_samples = 0
        with torch.no_grad():
            with tqdm(validation_dataloader, unit="batch") as tepoch:
                for states_batch, rewards_batch in tepoch:
                    # states_batch: (bsize, 150, 100)
                    # rewards_batch: (bsize, 150)
                    tepoch.set_description("Epoch {epoch} (validation)")
                    # In effect, we now batch by batch *and* t
                    trajectory_batch = states_batch.unsqueeze(1).expand(-1, 150, 150, 100) #repeat trajectory 150 times
                    trajectory_batch = trajectory_batch.reshape(-1, 150, 100) # return to batch form: (bsize*L, L, 100)
                    states_batch = states_batch.view(-1, 100) # (bsize*L, 100)
                    rewards_batch = rewards_batch.view(-1) # bsize * L
                    prediction = net.forward(trajectory_batch, states_batch)
                    loss = loss_func(prediction, rewards_batch)
                    avg_loss += loss.item() * args.batch_size * 150
                    n_samples += args.batch_size
                    tepoch.set_postfix({"avg validation loss": avg_loss/n_samples})
        avg_loss /= len(validation_dataloader)*150*args.batch_size
        wandb.log({'average validation loss': avg_loss})
    if not args.evaluate:
        torch.save(net.state_dict(), args.save_to)

if __name__ == "__main__":
    args = parse_args()
    train(args)