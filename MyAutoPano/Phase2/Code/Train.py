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
from Network.Network import UnSupHomographyModel
from Network.Network import SupHomographyModel
from Network.Network import UnsupLossFn
from Network.Network import SupLossFn

def apply_homography_and_crop_patch(I1, max_warp=32):
    h, w = I1.shape[:2]
    perturbations = np.float32([
        [np.random.uniform(-max_warp, max_warp), np.random.uniform(-max_warp, max_warp)],
        [np.random.uniform(-max_warp, max_warp), np.random.uniform(-max_warp, max_warp)],
        [np.random.uniform(-max_warp, max_warp), np.random.uniform(-max_warp, max_warp)],
        [np.random.uniform(-max_warp, max_warp), np.random.uniform(-max_warp, max_warp)],
    ])
    
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = src_points + perturbations
    
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    
    I1_warped = cv2.warpPerspective(I1, H, (w, h))
    
    return I1_warped, H

def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    I1WarpedBatch = []
    HomographyBatch = []
    
    for _ in range(MiniBatchSize):
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        RandImageName = os.path.join(BasePath, DirNamesTrain[RandIdx] + ".jpg")
        
        I1 = cv2.imread(RandImageName).astype(np.float32)
        I1 = cv2.resize(I1, (ImageSize[0], ImageSize[1]))
        
        I1_warped, H = apply_homography_and_crop_patch(I1)
        
        I1Batch.append(torch.from_numpy(I1.transpose(2, 0, 1)))
        I1WarpedBatch.append(torch.from_numpy(I1_warped.transpose(2, 0, 1)))
        HomographyBatch.append(torch.from_numpy(H))
    
    return torch.stack(I1Batch), torch.stack(I1WarpedBatch), torch.stack(HomographyBatch)

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


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
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
    if ModelType == 'Sup':
        # Predict output with forward pass
        model = SupHomographyModel(ImageSize, NumClasses)
        model.to(device)
    else:
        model = UnSupHomographyModel(ImageSize, NumClasses)
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

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
        
            # I1Batch, CoordinatesBatch= GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize)
            
            I1Batch, I1WarpedBatch, HomographyBatch = GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize)
            I1Batch = I1Batch.to(device)
            I1WarpedBatch = I1WarpedBatch.to(device)
            HomographyBatch = HomographyBatch.to(device)
            
            # if ModelType == 'Sup':
            #     # I1Batch, CoordinatesBatch= SupGenerateBatch(PerEpochCounter, BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize)
            #     PredicatedCoordinatesBatch = model(I1Batch)
            #     LossThisBatch = SupLossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            # else:
            #     # I1Batch, CoordinatesBatch = UnSupGenerateBatch(PerEpochCounter, BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize)
            #     PredicatedCoordinatesBatch = model(I1Batch, CoordinatesBatch)
            #     LossThisBatch = UnsupLossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            
            if ModelType == 'Sup':
                predicted_coordinates_batch = model(I1Batch, I1WarpedBatch)
                LossThisBatch = SupLossFn(predicted_coordinates_batch, HomographyBatch)
            else:
                predicted_coordinates_batch = model(I1Batch, I1WarpedBatch, HomographyBatch)
                LossThisBatch = UnsupLossFn(predicted_coordinates_batch, HomographyBatch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

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

            result = model.validation_step(I1Batch)
            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
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
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=2,
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
        default=1,
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
        default="Logs/",
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

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
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
    )


if __name__ == "__main__":
    main()
