#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
# from Network.Network import HomographyModel
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
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Network.Network import SupHomographyModel
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from Network.Network import UnSupHomographyModel

def Sup_GenerateBatch(PA_Path, PB_Path, H4Pt_Path, MiniBatchSize):
    PA_Batch = []
    PB_Batch = []
    H4Pt_Batch = []
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        RandIdx = random.randint(1, 5000)
        PA_RandImageName = f"{PA_Path}PA_{RandIdx}.jpg"
        PB_RandImageName = f"{PB_Path}PB_{RandIdx}.jpg"
        H4Pt_RandName = f"{H4Pt_Path}H4Pt_{RandIdx}.csv"

        PA = cv2.imread(PA_RandImageName)
        PB = cv2.imread(PB_RandImageName)
        if PA is None or PB is None:
            continue  # Skip this iteration if images are not found

        PA = (np.float32(PA) / 255.0).transpose(2, 0, 1)  # Normalize and transpose
        PB = (np.float32(PB) / 255.0).transpose(2, 0, 1)

        PA_Batch.append(torch.from_numpy(PA))
        PB_Batch.append(torch.from_numpy(PB))
        H4Pt = np.loadtxt(H4Pt_RandName, delimiter=',').reshape(-1).astype(np.float32)  # Ensure correct shape and type
        H4Pt_Batch.append(torch.from_numpy(H4Pt))

        ImageNum += 1

    return torch.stack(PA_Batch), torch.stack(PB_Batch), torch.stack(H4Pt_Batch)

def UnSup_GenerateBatch(PA_Path, PB_Path, CA_Path, MiniBatchSize):
    PA_Batch = []
    PB_Batch = []
    CA_Batch = []
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        RandIdx = random.randint(1, 5000)
        PA_RandImageName = f"{PA_Path}PA_{RandIdx}.jpg"
        PB_RandImageName = f"{PB_Path}PB_{RandIdx}.jpg"
        CA_RandName = f"{CA_Path}CA_{RandIdx}.csv"

        PA = cv2.imread(PA_RandImageName)
        PB = cv2.imread(PB_RandImageName)

        PA = (np.float32(PA) / 255.0).transpose(2, 0, 1)
        PB = (np.float32(PB) / 255.0).transpose(2, 0, 1)

        PA_Batch.append(torch.from_numpy(PA))
        PB_Batch.append(torch.from_numpy(PB))
        CA = np.loadtxt(CA_RandName, delimiter=',').reshape(-1).astype(np.float32)
        CA_Batch.append(torch.from_numpy(CA))

        ImageNum += 1

    return torch.stack(PA_Batch), torch.stack(PB_Batch), torch.stack(CA_Batch)

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def Sup_TrainOperation(
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
    NumClasses,
    CA_Path,
    H4Pt_path,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Predict output with forward pass
    model = SupHomographyModel()
    model.to(device)
        
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = torch.optim.AdamW(model.parameters(),lr = 0.0001)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        Optimizer.load_state_dict(CheckPoint["optimizer_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    PA_path = BasePath + "/PA/"
    PB_path = BasePath + "/PB/"
    
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        epoch_losses = []  # List to store all losses for the current epoch

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            PA_Batch, PB_Batch, H4Pt_Batch = Sup_GenerateBatch(PA_path, PB_path, H4Pt_path, MiniBatchSize)
            PA_Batch = PA_Batch.to(device)
            PB_Batch = PB_Batch.to(device)
            H4Pt_Batch = H4Pt_Batch.to(device)

            predicted_H4Pt_batch = model.forward(PA_Batch, PB_Batch)
            LossThisBatch = model.training_step(predicted_H4Pt_batch, H4Pt_Batch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            epoch_losses.append(LossThisBatch.item())  # Append the scalar loss

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                SaveName = CheckPointPath + str(Epochs) + "a" + str(PerEpochCounter) + "model.ckpt"
                torch.save({
                    "epoch": Epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(),
                    "loss": LossThisBatch,
                }, SaveName)
                print("\n" + SaveName + " Model Saved...")

        # Calculate the average loss for the epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Log the average epoch loss to tensorboard
        Writer.add_scalar("LossPerEpoch", avg_epoch_loss, Epochs)
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")

def UnSup_TrainOperation(
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
    NumClasses,
    CA_Path,
    H4Pt_path,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Predict output with forward pass
    model = UnSupHomographyModel()
    model.to(device)
        
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = torch.optim.AdamW(model.parameters(),lr = 0.0001)
    # Optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
    
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        Optimizer.load_state_dict(CheckPoint["optimizer_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    PA_path = BasePath + "/PA/"
    PB_path = BasePath + "/PB/"
    
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        epoch_losses = []  # List to store all losses for the current epoch

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            PA_Batch, PB_Batch, CA_Batch = UnSup_GenerateBatch(PA_path, PB_path, CA_Path, MiniBatchSize)
            PA_Batch = PA_Batch.to(device)
            PB_Batch = PB_Batch.to(device)
            CA_Batch = CA_Batch.to(device)

            predicted_H4Pt_batch = model.forward(PA_Batch, PB_Batch)
            LossThisBatch = model.training_step(CA_Batch, predicted_H4Pt_batch, PA_Batch, PB_Batch)

            LossThisBatch.requires_grad = True

            # Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            epoch_losses.append(LossThisBatch.item())  # Append the scalar loss

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                SaveName = CheckPointPath + str(Epochs) + "a" + str(PerEpochCounter) + "model.ckpt"
                torch.save({
                    "epoch": Epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(),
                    "loss": LossThisBatch,
                }, SaveName)
                print("\n" + SaveName + " Model Saved...")

        # Calculate the average loss for the epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Log the average epoch loss to tensorboard
        Writer.add_scalar("LossPerEpoch", avg_epoch_loss, Epochs)
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")

        
def plot_loss(logs_path):
    event_acc = EventAccumulator(logs_path)
    event_acc.Reload() 
    losses = event_acc.Scalars('LossPerEpoch')

    epochs = [entry.step for entry in losses]
    avg_epoch_losses = [entry.value for entry in losses]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Epoch Loss')
    plt.grid(True)
    plt.show()
    
def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    
    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and UnSup, Default:Sup",
    )
    ModelType = Parser.parse_args().ModelType
    
    Parser.add_argument(
        "--BasePath",
        default="./data/train",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default=f"../Checkpoints/{ModelType}/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=20,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=32,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default=f"./Logs/{ModelType}/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    create_dir_if_not_exists(CheckPointPath)
    create_dir_if_not_exists(LogsPath)
    
    NumTrainSamples = 5000
    SaveCheckPoint = 100
    NumClasses = 10
    ImageSize = [128, 128, 3]
    CA_Path = BasePath + "/CA/"
    H4Pt_path = BasePath + "/H4Pt/"
    
    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    if ModelType == "Sup":
        Sup_TrainOperation(
            NumTrainSamples,
            ImageSize,
            NumEpochs,
            MiniBatchSize,
            SaveCheckPoint,
            CheckPointPath,
            DivTrain,
            LatestFile,
            BasePath,
            LogsPath,
            ModelType,
            NumClasses,
            CA_Path,
            H4Pt_path,
        )
    elif ModelType == "UnSup":
        UnSup_TrainOperation(
            NumTrainSamples,
            ImageSize,
            NumEpochs,
            MiniBatchSize,
            SaveCheckPoint,
            CheckPointPath,
            DivTrain,
            LatestFile,
            BasePath,
            LogsPath,
            ModelType,
            NumClasses,
            CA_Path,
            H4Pt_path,
        )
    
    plot_loss(LogsPath)
        

if __name__ == "__main__":
    main()
