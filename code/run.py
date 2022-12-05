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
#from livelossplot.inputs.keras import PlotLossesCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '--task',
    #     required=True,
    #     choices=['1', '3'],
    #     help='''Which task of the assignment to run -
    #     training from scratch (1), or fine tuning VGG-16 (3).''')
    # parser.add_argument(
    #     '--data',
    #     default='..'+os.sep+'data'+os.sep,
    #     help='Location where the dataset is stored.')
    # parser.add_argument(
    #     '--load-vgg',
    #     default='vgg16_imagenet.h5',
    #     help='''Path to pre-trained VGG-16 file (only applicable to
    #     task 3).''')
    # parser.add_argument(
    #     '--load-checkpoint',
    #     default=None,
    #     help='''Path to model checkpoint file (should end with the
    #     extension .h5). Checkpoints are automatically saved when you
    #     train your model. If you want to continue training from where
    #     you left off, this is how you would load your weights.''')
    # parser.add_argument(
    #     '--confusion',
    #     action='store_true',
    #     help='''Log a confusion matrix at the end of each
    #     epoch (viewable in Tensorboard). This is turned off
    #     by default as it takes a little bit of time to complete.''')
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
    # parser.add_argument(
    #     '--lime-image',
    #     default='test/Bedroom/image_0003.jpg',
    #     help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


# def LIME_explainer(model, path, preprocess_fn):
#     """
#     This function takes in a trained model and a path to an image and outputs 5
#     visual explanations using the LIME model
#     """

#     def image_and_mask(title, positive_only=True, num_features=5,
#                        hide_rest=True):
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0], positive_only=positive_only,
#             num_features=num_features, hide_rest=hide_rest)
#         plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
#         plt.title(title)
#         plt.show()

#     # Read the image and preprocess it as before
#     image = imread(path)
#     if len(image.shape) == 2:
#         image = np.stack([image, image, image], axis=-1)
#     image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
#     image = preprocess_fn(image)
    

#     explainer = lime_image.LimeImageExplainer()

#     explanation = explainer.explain_instance(
#         image.astype('double'), model.predict, top_labels=5, hide_color=0,
#         num_samples=1000)

#     # The top 5 superpixels that are most positive towards the class with the
#     # rest of the image hidden
#     image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
#                    hide_rest=True)

#     # The top 5 superpixels with the rest of the image present
#     image_and_mask("Top 5 with the rest of the image present",
#                    positive_only=True, num_features=5, hide_rest=False)

#     # The 'pros and cons' (pros in green, cons in red)
#     image_and_mask("Pros(green) and Cons(red)",
#                    positive_only=False, num_features=10, hide_rest=False)

#     # Select the same class explained on the figures above.
#     ind = explanation.top_labels[0]
#     # Map each explanation weight to the corresponding superpixel
#     dict_heatmap = dict(explanation.local_exp[ind])
#     heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
#     plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
#     plt.colorbar()
#     plt.title("Map each explanation weight to the corresponding superpixel")
#     plt.show()


# def train(model, datasets, checkpoint_path, logs_path, init_epoch):
#     """ Training routine. """

#     # Keras callbacks for training
#     callback_list = [
#         tf.keras.callbacks.TensorBoard(
#             log_dir=logs_path,
#             update_freq='batch',
#             profile_batch=0),
#         ImageLabelingLogger(logs_path, datasets),
#         CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
#     ]

#     # Include confusion logger in callbacks if flag set
#     if ARGS.confusion:
#         callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

#     # Begin training
#     model.fit(
#         x=datasets.train_data,
#         validation_data=datasets.test_data,
#         epochs=hp.num_epochs,
#         batch_size=None,
#         callbacks=callback_list,
#         initial_epoch=init_epoch,
#     )
def train(model, X, y, epochs, batch_size):
    #plot_loss_1 = PlotLossesCallback()

    # ModelCheckpoint callback - save best weights
    tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

    # EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')
    h = model.fit(X, y, epochs = epochs, batch_size = batch_size, callbacks=[tl_checkpoint_1, early_stop],verbose = 1)
    return h


def test(model, X): #, y, batch_size):
    """ Testing routine. """

    # Run model on test set
    # predictions = model.evaluate(
    #     x=X,
    #     # y
    #     # batch_size = batch_size
    #     verbose=1,
    # )
    model.load_weights('tl_model_v1.weights.best.hdf5') 
    # initialize the best trained weights
    preds = model.predict(x=X)
    predictions = [np.argmax(p) for p in preds]
    return predictions


# def main():
#     """ Main function. """

#     time_now = datetime.now()
#     timestamp = time_now.strftime("%m%d%y-%H%M%S")
#     init_epoch = 0

#     # If loading from a checkpoint, the loaded checkpoint's directory
#     # will be used for future checkpoints
#     if ARGS.load_checkpoint is not None:
#         ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

#         # Get timestamp and epoch from filename
#         regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
#         init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
#         timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

#     # If paths provided by program arguments are accurate, then this will
#     # ensure they are used. If not, these directories/files will be
#     # set relative to the directory of run.py
#     if os.path.exists(ARGS.data):
#         ARGS.data = os.path.abspath(ARGS.data)
#     if os.path.exists(ARGS.load_vgg):
#         ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

#     # Run script from location of run.py
#     os.chdir(sys.path[0])

#     datasets = Datasets(ARGS.data, ARGS.task)

#     if ARGS.task == '1':
#         model = YourModel()
#         model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
#         checkpoint_path = "checkpoints" + os.sep + \
#             "your_model" + os.sep + timestamp + os.sep
#         logs_path = "logs" + os.sep + "your_model" + \
#             os.sep + timestamp + os.sep

#         # Print summary of model
#         model.summary()
#     else:
#         model = VGGModel()
#         checkpoint_path = "checkpoints" + os.sep + \
#             "vgg_model" + os.sep + timestamp + os.sep
#         logs_path = "logs" + os.sep + "vgg_model" + \
#             os.sep + timestamp + os.sep
#         model(tf.keras.Input(shape=(224, 224, 3)))

#         # Print summaries for both parts of the model
#         model.vgg16.summary()
#         model.head.summary()

#         # Load base of VGG model
#         model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

#     # Load checkpoints
#     if ARGS.load_checkpoint is not None:
#         if ARGS.task == '1':
#             model.load_weights(ARGS.load_checkpoint, by_name=False)
#         else:
#             model.head.load_weights(ARGS.load_checkpoint, by_name=False)

#     # Make checkpoint directory if needed
#     if not ARGS.evaluate and not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path)

#     # Compile model graph
#     model.compile(
#         optimizer=model.optimizer,
#         loss=model.loss_fn,
#         metrics=["sparse_categorical_accuracy"])

#     if ARGS.evaluate:
#         test(model, datasets.test_data)

#         # TODO: change the image path to be the image of your choice by changing
#         # the lime-image flag when calling run.py to investigate
#         # i.e. python run.py --evaluate --lime-image test/Bedroom/image_003.jpg
#         path = ARGS.lime_image
#         LIME_explainer(model, path, datasets.preprocess_fn)
#     else:
#         train(model, datasets, checkpoint_path, logs_path, init_epoch)

def main():
    TRAIN_PATH = ('../data/train/')
    TEST_PATH = ('../data/test/')
    MY_TEST_PATH = ('../data/my_test/')
    training_data = Datasets(TRAIN_PATH, hp.img_size)
    X_train, y_train, train_labels = training_data.load_data()
    print('Our Training Labels: ')
    print(train_labels)
    print()
    print(X_train.shape)
    print(y_train.shape)

    # testing_data = Datasets(TEST_PATH, hp.img_size)
    # X_test, y_test, testing_labels = testing_data.load_data()

    testing_data = Datasets(MY_TEST_PATH, hp.img_size)
    X_test, y_test, testing_labels = testing_data.load_data()

    model = VGGModel()
    # model.vgg16.summary()
    # model.head.summary()

    # model.compile(
    #     optimizer=model.optimizer,
    #     loss=model.loss_fn,
    #     metrics=["sparse_categorical_accuracy"])
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])
    
    
    
    #didn't return history in train method
    # h = train(model, X_train, y_train, hp.num_epochs, hp.batch_size)
    # h = train(model, X_train, y_train, 5, hp.batch_size)

    #this evaluates our model
    if ARGS.evaluate:
        h = train(model, X_train, y_train, 5, hp.batch_size)
        testing_data = Datasets(TEST_PATH, hp.img_size)
        X_test, y_test, testing_labels = testing_data.load_data()
        preds = test(model, X_test)
        # pred = test(model, X_test, y_test, hp.batch_size)
        print('Test accuracy... = %.2f' % accuracy_score(y_test, preds))
    if ARGS.makePredictions:
        data_gen = ImageDataGenerator(rescale = 1.0/255)
        pred_gen = data_gen.flow_from_directory(MY_TEST_PATH, target_size = hp.img_size, color_mode = "grayscale", batch_size = hp.batch_size, class_mode = "categorical", shuffle = False)
        preds = test(model, X_test)
        preds = [testing_labels[l] for l in preds]
        files = pred_gen.filenames
        actual_label = [testing_labels[l] for l in pred_gen.classes]
        results = pd.DataFrame({"file": files, "predictions": preds, "actual label": actual_label})
        for i in range(len(files)):
            file, prediction, actual = results.loc[i]
            path = os.path.join(MY_TEST_PATH, file)
            img = cv2.imread(path)
            plt.imshow(img)
            plt.title("Our model predicted class: {} {} Actual class: {}".format(prediction, '\n', actual))
            plt.show()

    else:
        h = train(model, X_train, y_train, 5, hp.batch_size)
    
    #this is predicting our images

# Make arguments global
ARGS = parse_args()

main()