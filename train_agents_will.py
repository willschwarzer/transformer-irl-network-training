import stable_baselines3
# import numpy as np
# import argparse
# import psutil
# import wandb
# from wandb.integration.sb3 import WandbCallback
# from spaceinvaders.spaceinvaders_block_env_regression import *
# from new_block_env_new import *
from object_env import ObjectEnv

rng = np.random.default_rng()

def env_to_str(env):
    return "_".join([f'{reward}' for reward in env.block_rewards])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-threads', '-mt', default=50, type=int)
    parser.add_argument('--num-envs', '-ne', default=100, type=int)
    parser.add_argument('--num-timesteps', '-nt', default=100000, type=int)
    parser.add_argument('--env', type=str, default="rings")
    parser.add_argument('--alg', type=str, default="PPO")
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--wandb-int', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-rewards', default=5, type=int, help="only used for ring env") 
    args = parser.parse_args()
    args.save_dir == None if args.save_dir.lower() == "none" else args.save_dir
    return args

@ray.remote
def train_agent(env, num_timesteps, save_dir, wandb_int):
    st = env_to_str(env)
    print("Training", st, "norm", sum([x ** 2 for x in env.block_rewards]))
    if wandb_int:
        run = wandb.init(project="ring_rl", reinit=True, sync_tensorboard=True)
        model = stable_baselines3.PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
        model.learn(total_timesteps=num_timesteps, callback=WandbCallback(verbose=2))
        run.finish()
    else:
        model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=num_timesteps)
    if save_dir is not None:
        print("Saving...")
        model.save("models_regression/" + st)
    else:
        print("\n\n\n\n\n\n\n NOT SAVING \n\n\n\n\n\n\n\n...")

if __name__ == '__main__':
    args = parse_args()
    n_threads = min(args.max_threads, psutil.cpu_count()-1)
    ray.init(num_cpus=n_threads)
    # ray.init()
    print(ray.available_resources())
    print(psutil.cpu_count())
    rewards = rng.uniform(-1, 1, size=(args.num_envs, args.num_rewards))
    envs = []
    for reward_set in rewards:
        reward_set_normed = reward_set/(np.sqrt(np.sum(reward_set**2)))
        if "invaders" in args.env or "space" in args.env:
            envs.append(SpaceInvadersBlockEnv(reward_set_normed))
        elif "ring" in args.env or "object" in args.env:
            envs.append(NewBlockEnv(reward_set_normed))
        elif "grid" in args.env:
            envs.append(ObjectEnv(reward_set_normed))
    ray.get([train_agent.remote(env, args.num_timesteps, args.save_dir, args.wandb_int) for env in envs])