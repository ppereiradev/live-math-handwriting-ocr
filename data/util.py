from tensorflow.keras.datasets import mnist
import numpy as np


'''
This script is responsible for loading two different datasets i.e. kaggle A-Z dataset, and MMIST,
so later we are able to put them together as one dataset.
'''


def load_numbers():
    # The MNIST dataset is divided into train and test set, so we load them,
    # and then we group them together.
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])

    # return a 2-tuple of the MNIST data and labels
    return (data, labels)


def load_letters(path):

    data = []
    labels = []

    # loop over the dataset composed of letters (i.e. A-Z handwritten)
    for r in open(path):
        # parse the label and image from each row
        r = r.split(",")
        label = int(r[0])
        img = np.array([int(i) for i in r[1:]], dtype="uint8")

        # The images in this dataset are represented by shades of gray
        # and they have 784 pixels. We take these pixels and transform into
        # a 28x28 matrix.
        img = img.reshape((28, 28))

        # we add the label and the image in their respective lists.
        data.append(img)
        labels.append(label)

    # we transform the lists into numpy arrays.
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")

    return (data, labels)
