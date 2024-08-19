# Just a simple text file to check the CUDA availabilty,
# version of CUDA and name of the device to install CuDNN accordingly.

# Used for enabling the GPU for the computation

import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Used CUDA 11.7 windows version
