import numpy as np
from pynvml import *

def get_freest_gpu():
    nvmlInit()
    NUM_DEVICES = 4
    free_mems = []
    for i in range(NUM_DEVICES):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        free_mems.append(info.free)
    return np.argmax(free_mems)
    # mems = [torch.cuda.memory_allocated(i) for i in range(num_devices)]
    # mems2 = [torch.cuda.memory_reserved(i) for i in range(num_devices)]
    # return mems, mems2
    
def convert_chai_rollouts(rollouts, horizon, obs_size, dtype):
    states = np.zeros((len(rollouts), horizon, obs_size), dtype=dtype)
    rewards = np.zeros((len(rollouts), horizon), dtype=float)
    for idx, rollout in enumerate(rollouts):
        rollout_traj = rollout.obs[:-2]
        if len(rollout_traj.shape) == 3:
            rollout_traj = np.reshape(rollout_traj, (rollout_traj.shape[0], -1))
        states[idx] = rollout_traj
        rewards[idx] = rollout.rews[:-1]
    return states, rewards