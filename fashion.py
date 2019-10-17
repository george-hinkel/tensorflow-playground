import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


class FashionML:
    def __init__(self):
        pass

    def RunML(self):
        fashion_mnist = keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
        ])

        model.compile(optimizer='SGD',
                      loss=tf.keras.losses.mean_squared_error,
                      metrics=[tf.metrics.categorical_accuracy])

        model.fit(train_images, train_labels, epochs=5)

        model.evaluate(test_images, test_labels, verbose=2)
