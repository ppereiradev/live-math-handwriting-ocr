from model.resnet import ResNet
from data.util import load_numbers
from data.util import load_letters
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


'''
This script is responsible for training the ResNet model. 
This is script is based on Dr. Adrian Rosebrock's tutorial 
(OCR with Keras, TensorFlow, and Deep Learning - Part 1)
which can be followed in: 
https://www.pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/

ATTENTION: It is worth pointing out that I forgot of using a math symbols to train the ResNet,
so it only understands multiplications of a variable and a coefficient,
for instance: 7x, or 8c, or 9.
'''

# Setting the number of epochs, learning rate,
# and batch size, which are used to train the ResNet model.
# It is only 5 epochs, because it took to long to train in
# my computer. 5 epochs took around 12 hours to train.
# To achieve better results, increase the number of epochs.
# In Dr. Adrian Rosebrock's tutorial, the epochs are set to 50.
EPOCHS = 5
INIT_LR = 1e-1
BS = 128

# Time to load the kaggle A-Z and MNIST datasets,
# make sure you have downloaded the kaggle A-Z dataset,
# which is in the following link, and put it inside the data folder:
# https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format
print("[INFO] loading datasets...")
# considering that you downloaded the same dataset, i.e. A_Z Handwritten Data.csv
(azData, azLabels) = load_letters("data/A_Z Handwritten Data.csv")
(numberData, numberLabels) = load_numbers()

# the MNIST dataset is labeled from 0 to 9, so each label of A-Z dataset
# is shifted by 10 to ensure the A-Z characters are not incorrectly labeled
# as numbers. Before the shift azLabels was [ 0  0  0 ... 25 25 25], after the shift
# azLabes became [10 10 10 ... 35 35 35].
azLabels += 10

# here we join these two datasets
data = np.vstack([azData, numberData])
labels = np.hstack([azLabels, numberLabels])

# The dimension of the images in A-Z and MNIST datasets are 28x28 pixels.
# Here, we are going to transform them to 32x32 images,
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

# Here, Dr. Adrian Rosebrock added a channel to every image in the dataset,
# then he scaled the pixel intensities of the images
# from [0, 255] to [0, 1].
data = np.expand_dims(data, axis=-1)
data /= 255.0

# converting labels from integers to vectors
le = LabelBinarizer()

labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# Here, it is taking care of the skew of the data, then it loops
# over all classes and calculate the class weight
classTotals = labels.sum(axis=0)
classWeight = {}

for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# spliting data into train (80%) and test (20%)
# it was set random_state to have same results with the Dr. Adrian Rosebrock's tutorial
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.20, stratify=labels, random_state=42)

# bulding the augmentation, using an image generator
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest")

# compiling the ResNet model, using SGD as optimizer
print("[INFO] compiling model...")

opt = SGD(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
                     (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


# training the network
print("[INFO] training network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1)

# defining the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# evaluating the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))


# saving the model to disk
print("[INFO] serializing network...")
model.save("ResNet_OCR.model", save_format="h5")

# Building a plot of training process
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

images = []

# Here, Dr. Adrian Rosebrock is randomly selecting a few testing characters
# then, he classifies the image and put a text of the classification
# over the image, the green text means correct, the red one means incorrect.
# After that, he resizes the images from 32x32 to 96x96, so it is better for
# us to see.
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):

    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    image = (testX[i] * 255).astype("uint8")
    color = (0, 255, 0)

    if prediction[0] != np.argmax(testY[i]):
        color = (0, 0, 255)

    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)

    images.append(image)

# building the montage for the images
montage = build_montages(images, (96, 96), (7, 7))[0]

# showing the output montage
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)
