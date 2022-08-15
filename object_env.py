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
import torch
import torch.nn.functional as F

viewport_width = 5 # must be odd integer
large_number = 1000000000000
empty_block_multiplier = 3 # higher number = more empty blocks
REWARD_BOUND = 100
NUM_BLOCKS = 3 # number of blocks which can be assigned a type
MAX_MVT_DIST = 1
MIN_BLOCK_DIST = 0.01 # To bound reward
ENV_SIZE = 100
assert(viewport_width == int(viewport_width))
assert(viewport_width % 2 == 1)

viewport_iteration_arr = [i-viewport_width//2 for i in range(viewport_width)]

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NOOP = 4

class ObjectEnv(gym.Env):

    def eat(self, x, y):
        self.eaten_blocks.add((x, y))

    def __init__(self, block_rewards, state_rep_model=None, seed=0):
        super().__init__()
        assert(len(block_rewards) == (NUM_BLOCKS*(NUM_BLOCKS-1)//2))

        # if state_rep is not None:
        #     self.state_rep = state_rep
        # else:
        #     self.state_rep = self._get_dist_sums
        
        self.key = random.PRNGKey(seed)
        
        # Make a triangular-array-compatible weight vector to allow for efficient
        # computation later
        tril = jnp.tril(jnp.ones([NUM_BLOCKS, NUM_BLOCKS]), k=-1) # Lower triangle of matrix, not including main diag
        expanded_block_rewards = tril.at[jnp.nonzero(tril)].set(block_rewards)
        
        
        self.expanded_block_rewards = expanded_block_rewards
        self.state_rep_model = state_rep_model

        self.action_space = spaces.Box(low=-1, high=1, shape=(3, 2))
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3, 2))
        
        self._init_obs()

    def _init_obs(self):
        self.key, subkey = random.split(self.key)
        self.state = random.uniform(subkey, shape=(NUM_BLOCKS, 2), minval=-ENV_SIZE, maxval=ENV_SIZE)

    @jit
    def _next_state(self):
        state = self._next_state(state, action, state_bounds)
        state += action
        state = jnp.clip(state, state_bounds[0], state_bounds[1])
        return state
    
    @jit
    def _step(state, action, weights, state_bounds):
        state = self._next_state(state, action)
        reward = self._get_reward(state, weights)
    
    @jit
    def _get_reward(state, weights):
        # state: (num_blocks, 2)
        num_pairs = len(state)*(len(state)-1)//2
        obj_matrix_a = jnp.expand_dims(state, 1)
        obj_matrix_b = jnp.expand_dims(state, 0)
        squared_diffs = jnp.pow(obj_matrix_a - obj_matrix_b, 2)
        squared_dists = jnp.sum(squared_diffs, axis=-1)
        dists = jnp.sqrt(squared_dists)
        clipped_dists = jnp.clip(dists, a_min=MIN_BLOCK_DIST)
        inv_dists = 1/clipped_dists
        reward = jnp.sum(inv_dists*weights)
        return reward
        

    def step(self, action, alt_reward=[0,0,0], render=False):
        if self.state_rep_model is None:
            self.state, reward = self._step(self.state, self.action, self.expanded_block_rewards)
        else:
            self.state = self._next_state(self.state, self.action)
            state_rep = self.state_rep_model(self.state)

        self.nsteps += 1
        done = self.nsteps > 150

        if action != Action.NOOP:
            # Apply slight penalty for moving
            reward -= 0.005

        return self.state.flatten(), reward, done

    def reset(self):
        self._init_obs()
        return self._next_observation().flatten()

    def render(self, mode='human', close=False):
        if mode == 'rgb':
            return self._next_render()
        else:
            return self._next_observation()
