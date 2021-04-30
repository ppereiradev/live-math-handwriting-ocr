import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import time
from sympy.plotting import plot
from sympy import *
import sympy as syp

'''
This script is responsible for capture the image from webcam,
then it passess each frame in the ResNet to classify the characters.
To finish, it put a box around each characters and plot a chart,
using the sympy library.

This is script is based on Dr. Adrian Rosebrock's tutorial 
(OCR with Keras, TensorFlow, and Deep Learning - Part 2)
which can be followed in: 
https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/

ATTENTION: It is worth pointing out that I forgot of using a math symbols to train the ResNet,
so it only understands multiplications of a variable and a coefficient,
for instance: 7x, or 8c, or 9.
'''


# loading the handwriting OCR model
print("[INFO] loading ResNet model...")
model = load_model("ResNet_OCR.model")

# setting the source of video,
# it is also possible to use the path
# of a video, e.g. "/home/user/video/ex_video.MOV"
capture = cv2.VideoCapture(0)

while(1):
    # reading the stream of the webcam, and initializiting
    # a variable, and a coefficient
    ret, image = capture.read()
    x_letter = ''
    x_number = ''

    if ret:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # performing contours detection and sorting the
        # them from left-to-right
        edged = cv2.Canny(blurred, 30, 150)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]

        # creating a list of contour bounding boxes
        chars = []

        for c in cnts:
            # computing the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)

            roi = gray[y:y + h, x:x + w]

            thresh = cv2.threshold(roi, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            chars.append((padded, (x, y, w, h)))

        boxes = [b[1] for b in chars]
        chars = np.array([c[0] for c in chars], dtype="float32")

        preds = model.predict(chars)

        labelNames = "0123456789"
        labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        labelNames = [l for l in labelNames]

        for (pred, (x, y, w, h)) in zip(preds, boxes):
            i = np.argmax(pred)
            prob = pred[i]
            label = labelNames[i]

            print("[INFO] {} - {:.2f}%".format(label, prob * 100))

            # filtering some objects that are classified as characters
            if prob*100 > 45:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                if label.isalpha():
                    x_letter = syp.symbols(label)
                elif label.isnumeric():
                    x_number = int(label)

        if x_letter != '' and x_number != '':
            p1 = x_number*x_letter
            graph = syp.plot(p1, show=False)
        elif x_letter == '' and x_number != '':
            p1 = x_number
            graph = syp.plot(p1, ylim=(-(x_number+1), x_number+1), show=False)
        elif x_letter != '' and x_number == '':
            p1 = x_letter
            graph = syp.plot(p1, show=False)
        elif x_letter == '' and x_number == '':
            p1 = 0
            graph = syp.plot(p1, show=False)

        graph.save('p1.png')

        p1_img = cv2.imread("p1.png")

        scale_percent = 50  # percent of original size
        width = int(p1_img.shape[1] * scale_percent / 100)
        height = int(p1_img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        p1_img = cv2.resize(p1_img, dim, interpolation=cv2.INTER_AREA)

        # putting chart on top-left corner, using ROI
        rows, cols, channels = p1_img.shape
        roi = image[0:rows, 0:cols]

        # creating a mask
        img2gray = cv2.cvtColor(p1_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(p1_img, p1_img, mask=mask)

        dst = cv2.add(img1_bg, img2_fg)
        image[0:rows, 0:cols] = dst

        cv2.imshow('image', image)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


capture.release()
cv2.destroyAllWindows()
