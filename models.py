from modules import *
from set_transformer.models import SetTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from object_env import _get_reward_features_torch

class NonLinearNet(nn.Module):
    def __init__(self, demonstration_rep_dim, 
                 state_rep_dim, 
                 state_hidden_size, 
                 reward_hidden_size, 
                 demonstration_hidden_size,
                 obs_size, 
                 horizon,
                 num_demonstration_layers, 
                 num_state_layers, 
                 dem_encoder_type='transformer',
                 mlp=False, 
                 demonstration_sigmoid=False, 
                 ground_truth_phi=False):
        
        super().__init__()
        assert demonstration_rep_dim == state_rep_dim, "Non-matching rep dims not yet implemented!"
        if dem_encoder_type == 'lstm':
            self.demonstration_encoder = DemonstrationNetLSTM(obs_size, num_demonstration_layers)
        elif dem_encoder_type == 'transformer':
            self.demonstration_encoder = DemonstrationNetTransformer(obs_size, horizon, num_demonstration_layers, demonstration_hidden_size, demonstration_rep_dim, demonstration_sigmoid)
        elif dem_encoder_type == 'set_transformer':
            self.demonstration_encoder = SetTransformer(obs_size, 1, demonstration_rep_dim)
        else:
            raise ValueError("Invalid demonstration encoder type!")
        # self.demonstration_encoder = SIDemonstrationNet()
        self.state_encoder = StateNet(state_rep_dim, state_hidden_size, num_state_layers, obs_size)
        self.mlp = mlp
        self.horizon = horizon
        if mlp:
            self.reward_layer = RewardNet(demonstration_rep_dim, state_rep_dim, reward_hidden_size)
        else:
            assert(demonstration_rep_dim == state_rep_dim)
        self.ground_truth_phi = ground_truth_phi
        if ground_truth_phi:
            # This was for the 5x5 gridworld
            # self.inv_distances = torch.zeros((5, 5)).float().cuda()
            # for x, y in itertools.product(range(5), repeat=2):
            #     if x == 2 and y == 2:
            #         continue
            #     self.inv_distances[x, y] = 1/(np.abs(x-2) + np.abs(y-2))
            # self.inv_distances[2, 2] = 20.
            # self.inv_distances = torch.flatten(self.inv_distances)

            # For the rings environment:
            self.MIN_OBJECT_DIST = 0.25

    
    def forward(self, demonstrations, states, weights=None):
        # demonstrations: (bsize, L, |S|) or (bsize, n, L, |S|)
        if weights is None:
            demonstration_rep = self.demonstration_encoder(demonstrations).squeeze() # (bsize, rep_dim)
        else:
            demonstration_rep = weights.squeeze() # (bsize, rep_dim)
        # states = demonstrations.view(-1, demonstrations.shape[-1]) # (bsize*L, |S|)
        if not self.ground_truth_phi:
            state_rep = self.state_encoder(states) # (bsize*L, rep_dim)
        else:
            # This was for the 5x5 gridworld
            # states_unflattened = states.view(-1, 25, 4)
            # inv_distances_unsqueezed = self.inv_distances.unsqueeze(0).unsqueeze(-1)
            # inv_distances_expanded = inv_distances_unsqueezed.expand(states.shape[0], -1, 4)
            # state_rep_with_empties = torch.einsum('bst, bst -> bt', states_unflattened, inv_distances_expanded) # Multiply everything then sum across the 25 squares (s)
            # state_rep = state_rep_with_empties[:, :3]

            # For the rings environment:
            # breakpoint()
            state_rep = _get_reward_features_torch(states, self.MIN_OBJECT_DIST, True) # (bsize, num_states, 5, 5)
            # tril = torch.tril(torch.ones([5, 5]), diagonal=-1).cuda()
            # expanded_demonstration_rep = torch.zeros_like(tril)
            # expanded_demonstration_rep[torch.nonzero(tril)] = expanded_demonstration_rep
            # reward = torch.sum(state_rep*expanded_demonstration_rep, dim=(2, 3))
        # If the batch size was 1, unsqueeze demonstration_rep to make it 2D
        breakpoint()
        if demonstrations.shape[0] == 1:
            demonstration_rep = demonstration_rep.unsqueeze(0)
        demonstration_rep_expanded = demonstration_rep.unsqueeze(1).expand(-1, self.horizon, -1) # (bsize, L, rep_dim)
        demonstration_rep_flattened = demonstration_rep_expanded.reshape(-1, demonstration_rep_expanded.shape[-1]) # (bsize*L, rep_dim)
        if self.mlp:
            reward = self.reward_layer(demonstration_rep_flattened, state_rep)
        else:
            reward = torch.einsum('bs,bs->b', demonstration_rep_flattened, state_rep)
        return reward.view(demonstrations.shape[0], -1)