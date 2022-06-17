import torch
import numpy as np
from pynvml import *

def get_freest_gpu():
    num_devices = torch.cuda.device_count()
    nvmlInit()
    free_mems = []
    for i in range(num_devices):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        free_mems.append(info.free)
    return np.argmax(free_mems)
    # mems = [torch.cuda.memory_allocated(i) for i in range(num_devices)]
    # mems2 = [torch.cuda.memory_reserved(i) for i in range(num_devices)]
    # return mems, mems2