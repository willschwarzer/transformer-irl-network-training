from modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 ground_truth_phi=False):
        
        super().__init__()
        assert trajectory_rep_dim == state_rep_dim, "Non-matching rep dims not yet implemented!"
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
    
