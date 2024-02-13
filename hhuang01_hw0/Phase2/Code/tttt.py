from torchviz import make_dot
from logging import root
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as tf
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
# import Misc.ImageUtils as iu
from Network.Network import CIFAR10Model, DenseNet, ResNet18, VGGNet, resnext50
from Misc.MiscUtils import *
from Misc.DataUtils import *
import torch.nn as nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from torch.utils.tensorboard import SummaryWriter


# model = CIFAR10Model(InputSize=3,OutputSize=10)
# model = ResNet18()
# model = DenseNet()
model = resnext50()    

writer = SummaryWriter('runs/model_visualization')


# 创建一个随机数据张量来模拟一个输入批次
# 假设模型输入是3x224x224的图像
input_tensor = torch.rand((1, 3, 224, 224))

# 将模型及输入写入TensorBoard
# 注意: make_grid函数适用于图像数据，对于模型图可视化不是必需的
writer.add_graph(model, input_tensor)
writer.close()

# 打印信息，告诉用户如何加载TensorBoard
print("Run 'tensorboard --logdir=runs' to start TensorBoard.")