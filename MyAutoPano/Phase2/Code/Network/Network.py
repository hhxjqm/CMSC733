"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pytorch_lightning as pl
import cv2
import kornia.geometry.transform as geometry


# Don't generate pyc codes
sys.dont_write_bytecode = True

###############################################
# Supervised Approach
###############################################

def SupLossF(predicted, target):
    return F.mse_loss(predicted, target)

class SupHomographyModel(nn.Module):
    def __init__(self):
        super(SupHomographyModel, self).__init__()
        self.net = Net()

    def forward(self, PA, PB):
        return self.net(PA, PB)

    def training_step(self, predicted_H4Pt_batch, H4Pt_batch):
        loss = SupLossF(predicted_H4Pt_batch, H4Pt_batch)
        return loss

    def validation_step(self, predicted_H4Pt_batch, H4Pt_batch):
        loss = SupLossF(predicted_H4Pt_batch, H4Pt_batch)
        return {"loss": loss}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: [64, 64, 64]

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: [128, 32, 32]

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: [128, 16, 16]

        # Fully connected layers
        # After the final pooling layer, the size will be [128, 16, 16]
        # We need to flatten this to connect to a fully connected layer
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 8)  # We predict 8 values for the homography

    def forward(self, PA, PB):
        x = torch.cat((PA, PB), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    

###############################################
# Unsupervised Approach
###############################################

def UnSupLossF(CA, H4Pt, PA, PB):
    CB = CA + H4Pt
    H = torch.stack([dlt(ca, cb) for ca, cb in zip(CA, CB)])
    w_PA = torch.stack([stn(pa, h) for pa, h in zip(PA, H)])
    w_PA = w_PA.to('cuda')
    PB = PB.to('cuda')
    
    loss = torch.norm(w_PA - PB, p=1, dim=(1, 2, 3)).mean()
    
    # criterion = nn.L1Loss()
    # loss = criterion(w_PA, PB)
    return loss

class UnSupHomographyModel(pl.LightningModule):
    def __init__(self):
        super(UnSupHomographyModel, self).__init__()
        self.net = Net()

    def forward(self, patch_a, patch_b):
        predicted_H4Pt_batch =  self.net(patch_a, patch_b)
        return predicted_H4Pt_batch

    def training_step(self, CA_Batch, predicted_H4Pt_batch, PA_Batch, PB_Batch):
        loss = UnSupLossF(CA_Batch, predicted_H4Pt_batch, PA_Batch, PB_Batch)
        return loss

    def validation_step(self, CA_Batch, predicted_H4Pt_batch, PA_Batch, PB_Batch):
        loss = UnSupLossF(CA_Batch, predicted_H4Pt_batch, PA_Batch, PB_Batch)
        return {"loss": loss}

def dlt(CA, CB):
    A = np.zeros((8, 9))
    CA_np = CA.to('cpu')
    CA_np = CA_np.detach().numpy()
    CB_np = CB.to('cpu')
    CB_np = CB_np.detach().cpu().numpy()
    
    for i in range(4):
        x, y = CB_np[2*i], CB_np[2*i + 1]
        X, Y = CA_np[2*i], CA_np[2*i + 1]
        A[2*i] = [-X, -Y, -1, 0, 0, 0, x*X, x*Y, x]
        A[2*i + 1] = [0, 0, 0, -X, -Y, -1, y*X, y*Y, y]
    A_tilde = A[:, :8]
    b_tilde = -A[:, 8]
    h_tilde, _, _, _ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
    h = np.append(h_tilde, 1)
    H = h.reshape(3, 3)
    return torch.tensor(H, dtype=torch.float32)

def stn(PA, H):
    # print(type(PA))
    # print("---------------------")
    # print(type(H))
    # print(H.shape)
    PA_numpy = PA.to('cpu')
    PA_numpy = PA_numpy.detach().numpy()
    PA_numpy = np.transpose(PA_numpy, (1,2,0))
    H_numpy = H.to('cpu')
    H_numpy = H_numpy.detach().numpy()
    
    warped_image = cv2.warpPerspective(PA_numpy, H_numpy, dsize=(128, 128))
    
    warped_image = (np.float32(warped_image) / 255.0).transpose(2, 0, 1)
    # warped_image = np.transpose(warped_image, (2,0,1))
    
    warped_tensor = torch.tensor(warped_image, dtype=torch.float32)
    
    return warped_tensor
