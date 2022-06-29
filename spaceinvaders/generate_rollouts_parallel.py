import numpy as np
import stable_baselines3
import ray
import psutil
import argparse
from spaceinvaders_block_env_regression import *
import os

@ray.remote
def get_examples_regression(n_examples, non_linear=False):
    def string_to_rewards(st):
        return [float(num) for num in st.split("_")]

    models_directory = "models_regression"
    models = os.listdir(models_directory)
    examples = []
    print(f"Found {len(models)} models")
    for model_name in models:
        rewards = string_to_rewards(model_name)

        print("Loading", model_name)
        print("Rewards", rewards)
        env = SpaceInvadersBlockEnv(rewards)
        model = stable_baselines3.PPO.load(models_directory + "/" + model_name, env)

        print("Done")
        for _ in range(n_examples):
            data = []
            obs = env.reset()
            for i in range(150):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                data.append((obs, rewards))
            if non_linear:
                example = {"rewards": rewards, "data": data}
            else:
                assert False, "not ready yet"
                # example = {"rewards": rewards, "data": data}
                # example = {"block_types": block_types, "block_powers": block_powers, "data": np.array(data), "rgb_decode": env.rgb_decode}
            examples.append(example)
            print("\rExamples:", len(examples), end="")
        print("")
    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-threads', '-mt', default=16, type=int)
    parser.add_argument('--out', '-o', type=str, default='examples')
    parser.add_argument('--non-linear', '-nl', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-examples', '-ne', type=int, default=1000)
    args = parser.parse_args()
    return args

all_examples = []
args = parse_args()
n_threads = min(args.max_threads, psutil.cpu_count()-1)
ray.init(num_cpus=n_threads)
# ray.init()
print(ray.available_resources())
print(psutil.cpu_count())
print(args.non_linear)
n_examples_per_thread = [args.num_examples//n_threads for i in range(n_threads-1)]
n_examples_per_thread.append(args.num_examples - sum(n_examples_per_thread))
outs = ray.get([get_examples_regression.remote(n_examples, args.non_linear) for n_examples in n_examples_per_thread])

for out in outs:
    all_examples.extend(out)

np.save(args.out + '_' + str(len(all_examples)) + ".npy", all_examples)
