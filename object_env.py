import random
import json
import math
import gym
from gym import spaces
from enum import IntEnum
import numpy as np
import jax.numpy as jnp
from jax import random
from jax import jit
from functools import partial
# import torch
# import torch.nn.functional as F

REWARD_BOUND = 100
MAX_MVT_DIST = 1
MIN_BLOCK_DIST = 0.01 # To bound reward

class ObjectEnv(gym.Env):

    def eat(self, x, y):
        self.eaten_blocks.add((x, y))

    def __init__(self, block_rewards, state_rep_model=None, seed=0, env_size=10, num_blocks=3):
        super().__init__()
        assert(len(block_rewards) == (num_blocks*(num_blocks-1)//2))

        # if state_rep is not None:
        #     self.state_rep = state_rep
        # else:
        #     self.state_rep = self._get_dist_sums
        
        self.nsteps = 0
        
        self.key = random.PRNGKey(seed)
        
        # Make a triangular-array-compatible weight vector to allow for efficient
        # computation later
        tril = jnp.tril(jnp.ones([num_blocks, num_blocks]), k=-1) # Lower triangle of matrix, not including main diag
        expanded_block_rewards = tril.at[jnp.nonzero(tril)].set(block_rewards)
        
        self.state_bounds = jnp.array([-env_size, env_size], dtype=float)
        
        self.expanded_block_rewards = expanded_block_rewards
        self.state_rep_model = state_rep_model

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,))
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3, 2))
        
        self.state, self.key = _init_state(self.key, num_blocks, env_size)

#     @partial(jit, static_argnums=(0,))
#     def _init_state(self, key, num_blocks, env_size):
#         key, subkey = random.split(key)
#         state = random.uniform(subkey, shape=(num_blocks, 2), minval=-env_size, maxval=env_size)
#         return state, key
        
#     @partial(jit, static_argnums=(0,))
#     def _step(self, state, action, weights, state_bounds):
#         state = self._next_state(state, action, state_bounds)
#         reward = self._get_reward(state, weights)
#         return state, reward

#     @partial(jit, static_argnums=(0,))
#     def _next_state(self, state, action, state_bounds):
#         state += jnp.reshape(action, (len(action), 2))
#         state = jnp.clip(state, state_bounds[0], state_bounds[1])
#         return state
    
#     @partial(jit, static_argnums=(0,))
#     def _get_reward(self, state, weights):
#         # state: (num_blocks, 2)
#         num_pairs = len(state)*(len(state)-1)//2
#         obj_matrix_a = jnp.expand_dims(state, 1)
#         obj_matrix_b = jnp.expand_dims(state, 0)
#         squared_diffs = jnp.power(obj_matrix_a - obj_matrix_b, 2)
#         squared_dists = jnp.sum(squared_diffs, axis=-1)
#         dists = jnp.sqrt(squared_dists)
#         clipped_dists = jnp.clip(dists, a_min=MIN_BLOCK_DIST)
#         inv_dists = 1/clipped_dists
#         reward = jnp.sum(inv_dists*weights)
#         return reward
        

    def step(self, action, alt_reward=[0,0,0], render=False):
        action = jnp.array(action)
        if self.state_rep_model is None:
            self.state, reward = _step(self.state, action, self.expanded_block_rewards, self.state_bounds)
        else:
            self.state = _next_state(self.state, action, self.state_bounds)
            state_rep = state_rep_model(self.state)

        self.nsteps += 1
        done = self.nsteps > 150

        return self.state, reward, done, {}

    def reset(self):
        self.state, self.key = _init_state(self.key, self.num_blocks, self.env_size)
        return self.state

    def render(self, mode='human', close=False):
        if mode == 'rgb':
            return self._next_render()
        else:
            return self._next_observation()

@partial(jit, static_argnums=(1, 2))
def _init_state(key, num_blocks, env_size):
    key, subkey = random.split(key)
    state = random.uniform(subkey, shape=(num_blocks, 2), minval=-env_size, maxval=env_size)
    return state, key

@partial(jit, static_argnums=(3))
def _step(state, action, weights, state_bounds):
    state = _next_state(state, action, state_bounds)
    reward = _get_reward(state, weights)
    return state, reward

@partial(jit, static_argnums=(2))
def _next_state(state, action, state_bounds):
    state += jnp.reshape(action, (len(action), 2))
    state = jnp.clip(state, state_bounds[0], state_bounds[1])
    return state

@partial(jit, static_argnums=())
def _get_reward(state, weights):
    # state: (num_blocks, 2)
    num_pairs = len(state)*(len(state)-1)//2
    obj_matrix_a = jnp.expand_dims(state, 1)
    obj_matrix_b = jnp.expand_dims(state, 0)
    squared_diffs = jnp.power(obj_matrix_a - obj_matrix_b, 2)
    squared_dists = jnp.sum(squared_diffs, axis=-1)
    dists = jnp.sqrt(squared_dists)
    clipped_dists = jnp.clip(dists, a_min=MIN_BLOCK_DIST)
    inv_dists = 1/clipped_dists
    reward = jnp.sum(inv_dists*weights)
    return reward