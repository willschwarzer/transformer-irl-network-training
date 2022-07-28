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