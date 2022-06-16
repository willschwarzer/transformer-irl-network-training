print("NOTE run with python3.6 on base conda environment in october_miniconda_install")

import numpy as np
import stable_baselines3
from spaceinvaders_block_env import *

examples = []
for npos in range(4):
    for nneu in range(4-npos):
        nneg = 3-npos-nneu

        listt = []
        st = ""
        for i in range(npos):
            listt.append(BlockType.POSITIVE)
            st += "positive_"
        for i in range(nneu):
            listt.append(BlockType.NEUTRAL)
            st += "neutral_"
        for i in range(nneg):
            listt.append(BlockType.NEGATIVE)
            st += "negative_"
        st = st[:-1]

        print("Loading", st)
        env = SpaceInvadersBlockEnv(listt)
        model = stable_baselines3.PPO.load("models/" + st, env)

        print("Done")
        for _ in range(1000): #10000
            example_obs = []
            obs = env.reset()
            for i in range(150):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                example_obs.append(env.render(mode='rgb').flatten())
            example = {"key": listt, "data": np.array(example_obs), "rgb_decode": env.rgb_decode}
            examples.append(example)
            print("\rExamples:", len(examples), end="")
        print("")

np.save("examples_" + str(len(examples)) + ".npy", examples)
