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
        # Concatenate PA and PB along the channel dimension
        x = torch.cat((PA, PB), dim=1)  # [batch_size, 6, 128, 128]
        
        # Forward pass through the convolutional layers and pooling layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 16 * 16)  # Flatten the tensor
        
        # Forward pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation function, because this is a regression problem
        
        return x
    
    

###############################################
# Unsupervised Approach
###############################################

def UnSupLossF(CA, H4Pt, PA, PB):
    CB = CA + H4Pt
    H = torch.stack([dlt(ca, cb) for ca, cb in zip(CA, CB)])
    w_PA = torch.stack([stn(pa, h) for pa, h in zip(PA, H)])
    loss = torch.norm(w_PA - PB, p=1, dim=(1, 2, 3)).mean()
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
    CA = CA.detach().cpu().numpy()
    CB = CB.detach().cpu().numpy()
    for i in range(4):
        x, y = CB[2*i], CB[2*i + 1]
        X, Y = CA[2*i], CA[2*i + 1]
        A[2*i] = [-X, -Y, -1, 0, 0, 0, x*X, x*Y, x]
        A[2*i + 1] = [0, 0, 0, -X, -Y, -1, y*X, y*Y, y]
    A_tilde = A[:, :8]
    b_tilde = -A[:, 8]
    h_tilde, _, _, _ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)
    h = np.append(h_tilde, 1)
    H = h.reshape(3, 3)
    return torch.tensor(H, dtype=torch.float32)



# Define your stn function which stacks transformations for each pair of PA and H
def stn(PA, H):
    
    PA_numpy = PA.detach().cpu().numpy()
    H_numpy = H.detach().cpu().numpy()
    
    print("PPPPPA_numpy: ")
    print(PA_numpy.shape)
    print("H_numpy: ")
    print(H_numpy.shape)
    
    warped_image = cv2.warpPerspective(PA_numpy, H_numpy, dsize=(128, 128))
    print("warped_image: ")
    print(warped_image.shape)
    
    warped_tensor = torch.tensor(warped_image, dtype=torch.float32)
    exit(0)
    # grid = F.affine_grid(H, PA, align_corners=False)
    # warped_image = F.grid_sample(PA, grid, align_corners=False)
    
    return warped_tensor


# def dlt_svd(CA, CB):
#     if CA.shape != (4, 2) or CB.shape != (4, 2):
#         raise ValueError("CA and CB must be 4x2 arrays representing four points.")
#     N = 4
#     A = np.zeros((2 * N, 9))
#     for i in range(N):
#         X = CA[i, 0]
#         Y = CA[i, 1]
#         x = CB[i, 0]
#         y = CB[i, 1]
#         A[2 * i] = [-X, -Y, -1, 0, 0, 0, x * X, x * Y, x]
#         A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, y * X, y * Y, y]
#     U, S, Vt = np.linalg.svd(A)
#     H = Vt[-1].reshape(3, 3)
#     H = H / H[-1, -1]
#     return H
