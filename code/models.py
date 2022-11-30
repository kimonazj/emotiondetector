import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # TASK 3
        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        # Don't change the below:

        #NEW EDITS
        self.vgg16 =  tf.keras.applications.vgg16(weights = None, include_top = False, input_shape = (48,48,3))

        # TASK 3
        # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
        #       pretrained VGG16 weights into place so that only the classificaiton
        #       head is trained.
        for i in range(len(self.vgg16)):
              self.vgg16[i].trainable = False

        # TODO: Write a classification head for our 15-scene classification task.
        self.head = [
              Dense(256, activation='relu'),
              Flatten(),
              Dropout(0.3),
              Dense(hp.num_classes, activation='softmax')
        ]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TASK 3
        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        return loss