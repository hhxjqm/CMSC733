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

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from torchvision import transforms as tf
from Network.Network import SupHomographyModel
from Network.Network import UnSupHomographyModel

# Don't generate pyc codes
sys.dont_write_bytecode = True

def load_model(model_path, ModelType):
    if ModelType == "Sup":
        model = SupHomographyModel()
    else:
        model = UnSupHomographyModel()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def process_images(pa_path, pb_path):
    PA = cv2.imread(pa_path)
    PB = cv2.imread(pb_path)
    PA = (np.float32(PA) / 255.0).transpose(2, 0, 1)
    PB = (np.float32(PB) / 255.0).transpose(2, 0, 1)
    return PA, PB

def predict_homography(model, PA, PB):
    PA_batch = [torch.from_numpy(PA)]
    PB_batch = [torch.from_numpy(PB)]
    predicted_H4Pt = model.forward(torch.stack(PA_batch), torch.stack(PB_batch))
    return predicted_H4Pt.detach().numpy().reshape(4, 2)

def load_points(ca_path, h4pt_path):
    CA = np.loadtxt(ca_path, delimiter=',')
    H4Pt = np.loadtxt(h4pt_path, delimiter=',')
    return CA, H4Pt

def add_offsets_to_corners(CA, offsets):
    return np.add(CA, offsets)

def draw_points_and_lines(image, points, color, thickness):
    int_points = np.int32(points)
    for point in int_points:
        cv2.circle(image, tuple(point), 5, color, -1)
    cv2.polylines(image, [int_points], isClosed=True, color=color, thickness=thickness)

def save_image(image_path, image):
    cv2.imwrite(image_path, image)

def get_random_indices(limit, count):
    return random.sample(range(1, limit+1), count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ModelType", default="Sup", help="Model type, Supervised or Unsupervised? Choose from Sup and UnSup, Default:Sup")
    ModelType = parser.parse_args().ModelType
    if ModelType == "Sup":
        parser.add_argument("--ModelPath", dest="ModelPath", default="../Checkpoints/Sup/9model.ckpt", help="Path to load latest model from, Default:ModelPath")
    elif ModelType == "UnSup":
        parser.add_argument("--ModelPath", dest="ModelPath", default="../Checkpoints/UnSup/9model.ckpt", help="Path to load latest model from, Default:ModelPath")
    else:
        exit("Wrong ModelType, pick Sup or UnSup")
    parser.add_argument("--BasePath", dest="BasePath", default="./data/test/", help="Path to load images from, Default:BasePath")
    parser.add_argument("--NumTest", dest="NumTest", default=5, help="Number of image you want to test, Default:5")

    
    args = parser.parse_args()
    ModelPath = args.ModelPath
    BasePath = args.BasePath
    NumTest = args.NumTest
    model = load_model(ModelPath, ModelType)
    
    
    random_indices = get_random_indices(1000, NumTest)
    for i in random_indices:
        pa_path = f'{BasePath}PA/PA_{i}.jpg'
        pb_path = f'{BasePath}PB/PB_{i}.jpg'
        ca_path = f'{BasePath}CA/CA_{i}.csv'
        h4pt_path = f'{BasePath}H4Pt/H4Pt_{i}.csv'
        image_save_path = f'../Result/{ModelType}/{i}_Pre_vs_Ori.jpg'

        PA, PB = process_images(pa_path, pb_path)
        predicted_H4Pt = predict_homography(model, PA, PB)
        CA, H4Pt = load_points(ca_path, h4pt_path)
        predicted_CB = add_offsets_to_corners(CA, predicted_H4Pt)
        CB = add_offsets_to_corners(CA, H4Pt)

        original_image = cv2.imread(f'../Data/Val/{i}.jpg')
        draw_points_and_lines(original_image, predicted_CB, (0, 0, 255), 2)
        draw_points_and_lines(original_image, CB, (255, 0, 0), 2)
        save_image(image_save_path, original_image)
        print("Images save in " + image_save_path)
        
if __name__ == "__main__":
    main()