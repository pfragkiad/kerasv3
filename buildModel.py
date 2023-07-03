
import tensorflow as tf
import numpy as np
import cv2

import os.path as path

def getTest():
    mnist = tf.keras.datasets.mnist  # 28x28 size images of hand-writter digits 0-9
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    return (xTest, yTest)

def getModel(filename = None):
    if path.exists(filename):
        return tf.keras.models.load_model(filename)

    print(f"Tensoflow version: {tf.__version__}")

    mnist = tf.keras.datasets.mnist  # 28x28 size images of hand-writter digits 0-9
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print(len(xTrain), len(yTrain), len(xTest), len(yTest))

    model = tf.keras.models.Sequential()

    model.add(    tf.keras.layers.Flatten()) #input_shape=(28,28))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    #we want to minimize loss
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(xTrain,yTrain,epochs=3)

    loss, accuracy = model.evaluate(xTest, yTest)
    print (f'Loss: {loss}, Accuracy: {accuracy}')

    model.save(filename)

    return model