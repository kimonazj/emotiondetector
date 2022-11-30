
"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import cv2

import hyperparameters as hp


class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, img_size):
        self.data_path = data_path
        self.img_size = img_size

    def load_data(self):
        X, y = [],[]
        i = 0
        labels = {}
        for path in tqdm(sorted(os.listdir(self.data_path))):
            if not path.startswith('.'):
                labels[i] = path
                for file in os.listdir(self.data_path + path):
                    if not file.startswith('.'):
                        img = cv2.imread(self.data_path + path + '/' + file)
                        img = img.astype('float32') / 255
                        resized = cv2.resize(img, self.img_size, interpolation = cv2.INTER_AREA)
                        X.append(resized)
                        y.append(i)
                i += 1
        X = np.array(X)
        y = np.array(y)
        print(f'{len(X)} images loaded from {self.data_path} directory.')
        return X, y, labels
