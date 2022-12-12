import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import VGGModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver
from sklearn.metrics import accuracy_score
from skimage.io import imread
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--makePredictions',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.'''
    )

    return parser.parse_args()

def train(model, X, y, epochs, batch_size):
    h = model.fit(X, y, epochs = epochs, batch_size = batch_size, verbose = 1)
    return h


def test(model, X): #, y, batch_size):
    """ Testing routine. """
    preds = model.predict(x=X)
    predictions = [np.argmax(p) for p in preds]
    return predictions

def main():
    TRAIN_PATH = ('../data/train/')
    TEST_PATH = ('../data/test/')
    MY_TEST_PATH = ('../data/my_test/')
    TESTY_PATH = ('../data/testy/')
    training_data = Datasets(TRAIN_PATH, hp.img_size)
    X_train, y_train, train_labels = training_data.load_data()
    print('Our Training Labels: ')
    print(train_labels)
    print()


    # testing_data = Datasets(MY_TEST_PATH, hp.img_size)
    testing_data = Datasets(TESTY_PATH, hp.img_size)
    X_test, y_test, testing_labels = testing_data.load_data()

    model = VGGModel()
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])
    
    h = train(model, X_train, y_train, hp.num_epochs, hp.batch_size)

    #this evaluates our model
    if ARGS.evaluate:
        testing_data = Datasets(TESTY_PATH, hp.img_size)
        X_test, y_test, testing_labels = testing_data.load_data()
        preds = test(model, X_test)
        # pred = test(model, X_test, y_test, hp.batch_size)
        print('Test accuracy... = %.2f' % accuracy_score(y_test, preds))
    elif ARGS.makePredictions:
        data_gen = ImageDataGenerator(rescale = 1.0/255)
        pred_gen = data_gen.flow_from_directory(TESTY_PATH, target_size = hp.img_size, color_mode = "grayscale", batch_size = hp.batch_size, class_mode = "categorical", shuffle = False)
        preds = test(model, X_test)
        preds = [testing_labels[l] for l in preds]
        files = pred_gen.filenames
        actual_label = [testing_labels[l] for l in pred_gen.classes]
        results = pd.DataFrame({"file": files, "predictions": preds, "actual label": actual_label})
        for i in range(len(files)):
            file, prediction, actual = results.loc[i]
            path = os.path.join(TESTY_PATH, file)
            img = cv2.imread(path)
            plt.imshow(img)
            plt.title("Our model predicted class: {} {} Actual class: {}".format(prediction, '\n', actual))
            plt.show()

# Make arguments global
ARGS = parse_args()

main()