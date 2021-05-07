import scipy.io as sio
import random
import numpy as np

# 数据集读取
mat = sio.loadmat('ORL_32_32.mat')
img = mat['alls']
label = mat['gnd']
print(img.size())