import tensorflow
import torch

if torch.cuda.is_available():
    print('torch.cuda-available')
print(torch.__version__)
print(torch.version.cuda)
if tensorflow.test.is_built_with_cuda():
    print('tf-cuda-available')
if tensorflow.test.is_gpu_available():
    print('tf-available')
tensorflow.config.list_physical_devices('GPU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")