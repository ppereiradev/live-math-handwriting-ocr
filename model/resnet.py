import tensorflow as tf
from tensorflow import keras


'''
This script is an implementation of a ResNet model, based on the tutorial of Ms. Gracelyn Shi,
which can be found in:
https://towardsdatascience.com/implementing-a-resnet-model-from-scratch-971be7193718

We adapted this script to use the Keras of Tensorflow implementation. According to Ms. Gracelyn Shi,
she followed an implementation of Dr. Adrian Rosebrock (Deep Learning for Computer Vision with Python (2017)). 
'''


class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
                        reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be
        # initialize as the keras.layers.Input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                              momentum=bnMom)(data)
        act1 = keras.layers.Activation("relu")(bn1)
        conv1 = keras.layers.Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(reg))(act1)

        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                              momentum=bnMom)(conv1)
        act2 = keras.layers.Activation("relu")(bn2)
        conv2 = keras.layers.Conv2D(int(K * 0.25), (3, 3), strides=stride,
                                    padding="same", use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1
        # CONVs
        bn3 = keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                              momentum=bnMom)(conv2)
        act3 = keras.layers.Activation("relu")(bn3)
        conv3 = keras.layers.Conv2D(K, (1, 1), use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(reg))(act3)

        # if we are to reduce the spatial size, apply a CONV layer to
        # the shortcut
        if red:
            shortcut = keras.layers.Conv2D(K, (1, 1), strides=stride,
                                           use_bias=False, kernel_regularizer=keras.regularizers.l2(reg))(act1)

        # keras.layers.add together the shortcut and the final CONV
        x = keras.layers.add([conv3, shortcut])

        # return the keras.layers.addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters,
              reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
        # initialize the keras.layers.Input shape to be "channels last" and the
        # channels dimension itself
        keras.layers.InputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the keras.layers.Input shape
        # and channels dimension
        if tf.keras.backend.image_data_format() == "channels_first":
            keras.layers.InputShape = (depth, height, width)
            chanDim = 1

        # set the keras.layers.Input and apply BN
        keras.layers.Inputs = keras.layers.Input(shape=keras.layers.InputShape)
        x = keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                            momentum=bnMom)(keras.layers.Inputs)

        # check if we are utilizing the CIFAR dataset
        if dataset == "cifar":
            # apply a single CONV layer
            x = keras.layers.Conv2D(filters[0], (3, 3), use_bias=False,
                                    padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)

        # check to see if we are using the Tiny ImageNet dataset
        elif dataset == "tiny_imagenet":
            # apply CONV => BN => ACT => POOL to reduce spatial size
            x = keras.layers.Conv2D(filters[0], (5, 5), use_bias=False,
                                    padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
            x = keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                                momentum=bnMom)(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.ZeroPadding2D(
                (1, 1))(x)
            x = keras.layers.MaxPooling2D(
                (3, 3), strides=(2, 2))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the keras.layers.Input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                                       chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        # apply BN => ACT => POOL
        x = keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                            momentum=bnMom)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(
            classes, kernel_regularizer=keras.regularizers.l2(reg))(x)
        x = keras.layers.Activation("softmax")(x)

        # create the keras.models.Model
        keras.models.Model = keras.models.Model(
            keras.layers.Inputs, x, name="resnet")

        # return the constructed network architecture
        return keras.models.Model
