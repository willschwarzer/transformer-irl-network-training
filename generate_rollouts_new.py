import numpy as np
import stable_baselines3
import imitation
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import save
import psutil
import argparse
import os
from spaceinvaders.spaceinvaders_block_env_regression import *
from new_block_env_new import *
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import ray

# Currently not using gpu until I tell it how many are required
@ray.remote
def get_examples_regression(model_name, num_examples, space_invaders, non_linear, verbose, chai_rollouts):
    # stupidest thing I've ever seen, this doesn't load if it's not in the ray remote
    from imitation.data import rollout
    def string_to_rewards(st):
        return np.array([float(num) for num in st.split("_")])

    weights = string_to_rewards(model_name)

    if verbose >= 1:
        print("Rewards", weights)
    if space_invaders:
        env = SpaceInvadersBlockEnv(weights)
    else:
        env = NewBlockEnv(weights)
    models_directory = "models_regression"
    model = stable_baselines3.PPO.load(models_directory + "/" + model_name, env)

    if chai_rollouts:
        rollouts = rollout.rollout(
    model,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=int(num_examples*1.3)),
)
        for rollout in rollouts:
            if rollout.obs.shape[0] != 152:
                rollouts.remove(rollout)
        assert len(rollouts) >= num_examples, "Too many bugged rollouts (L < 150)"
        rollouts = rollouts[:num_examples]
        return (rollouts, weights)
    else:
        examples = []
        for _ in range(num_examples):
            data = []
            obs = env.reset()
            for i in range(150):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                data.append((obs, rewards))
            if non_linear:
                example = {"data": data, "weights": weights}
            else:
                assert False, "not ready yet"
                # example = {"rewards": rewards, "data": data}
                # example = {"block_types": block_types, "block_powers": block_powers, "data": np.array(data), "rgb_decode": env.rgb_decode}
            examples.append(example)
            if verbose >= 1:
                print("\rExamples:", len(examples), end="")
        if verbose >= 1:
            print("")
        return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-threads', '-mt', default=50, type=int)
    parser.add_argument('--out', '-o', type=str, default='examples')
    parser.add_argument('--non-linear', '-nl', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-examples', '-ne', type=int, default=1000)
    parser.add_argument('--num-models', '-nm', type=int, default=0)
    parser.add_argument('--space-invaders', '-si', action='store_true')
    parser.add_argument('--verbose', '-v', type=int, default=0)
    parser.add_argument('--chai-rollouts', '-cr', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

all_examples = []
args = parse_args()
n_threads = min(args.max_threads, psutil.cpu_count()-1)
ray.init(num_cpus=n_threads)
# ray.init()
print(ray.available_resources())
print(psutil.cpu_count())
models_directory = "models_regression"
models = os.listdir(models_directory)
print(f"Found {len(models)} models")
num_models = len(models) if args.num_models == 0 else args.num_models
outs = ray.get([get_examples_regression.remote(model_name, args.num_examples, args.space_invaders, args.non_linear, args.verbose, args.chai_rollouts) for model_name in models[:num_models]])

if args.chai_rollouts:
    rollouts = [ex[0] for ex in outs]
    weights = [ex[1] for ex in outs]
    for rollout in rollouts:
        all_examples.extend(rollout)
    save("_".join([args.out, "chai", str(len(all_examples))]), all_examples)
    np.save("_".join(["weights", "chai", str(len(all_examples))]), weights)
else:
    for out in outs:
        all_examples.extend(outs)
    np.save(args.out + '_' + str(len(all_examples)) + ".npy", all_examples)