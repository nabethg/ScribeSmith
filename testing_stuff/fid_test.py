import os
import re
import csv
import datetime

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.spectral_norm import spectral_norm

import torch.optim as optim

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, Subset
from torchvision.io import read_image
from torchvision.transforms import Grayscale, Resize

from torchmetrics.image.fid import FrechetInceptionDistance
import torch_fidelity
import random
import time

a = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

fid = FrechetInceptionDistance(feature=2048)
fid.to(device)
imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist1 = imgs_dist1.to(device)
imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist2 = imgs_dist2.to(device)


fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
print(fid.compute())
print(time.time()-a)

# plt.show()

param_size = 0
for param in fid.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in fid.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))
