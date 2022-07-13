import stable_baselines3
import numpy as np
import ray
import argparse
import psutil
from spaceinvaders.spaceinvaders_block_env_regression import *
from new_block_env_new import *

rng = np.random.default_rng()

def env_to_str(env):
    return "_".join([f'{reward}' for reward in env.block_rewards])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-threads', '-mt', default=50, type=int)
    parser.add_argument('--num-envs', '-ne', default=100, type=int)
    parser.add_argument('--space-invaders', action='store_true')
    args = parser.parse_args()
    return args

@ray.remote
def train_agent(env):
    st = env_to_str(env)
    print("Training", st, "norm", sum([x ** 2 for x in env.block_rewards]))
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    print("Saving...")
    model.save("models_regression/" + st)
    print(st)

if __name__ == '__main__':
    args = parse_args()
    n_threads = min(args.max_threads, psutil.cpu_count()-1)
    ray.init(num_cpus=n_threads)
    # ray.init()
    print(ray.available_resources())
    print(psutil.cpu_count())
    rewards = rng.uniform(-1, 1, size=(args.num_envs, 3))
    envs = []
    for reward_set in rewards:
        reward_set_normed = reward_set/(np.sqrt(np.sum(reward_set**2)))
        if args.space_invaders:
            envs.append(SpaceInvadersBlockEnv(reward_set_normed))
        else:
            envs.append(NewBlockEnv(reward_set_normed))
    ray.get([train_agent.remote(env) for env in envs])