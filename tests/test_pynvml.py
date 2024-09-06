import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
result = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(result)

import torch
free_mem, total_mem = torch.cuda.mem_get_info()
print('free_mem, total_mem: ', free_mem, total_mem)