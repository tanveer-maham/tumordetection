import dippykit as dip
import numpy as np
import cv2
from Shape_Detection_And_Skull_Stripping import SkullAndShape
from Noise_Filtering import NoiseRemoval
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

## Define mode
mode = 4   # 0: Regular, 1: Skull Stripping, 2: Gaussian, 3: Median, 4: Anistropic, 5: Non-local


## Define Down Sampling Matrix
D = 8;
Down_Sampled_Matrix = np.array([[D, 0],
                                [0, D]]);


# Generate Training Set
Training_images = np.zeros([100, 64, 64])
Training_labels = np.zeros([100, 1])

if mode == 0:

    for i in range(50):

        path1 = "Tumor_Images/" + str(i+1) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 1) + ".png"
        Training_images[i, :, :] = dip.resample(dip.im_to_float(dip.im_read(path1)), Down_Sampled_Matrix)
        Training_images[i+50, :, :] = dip.resample(dip.im_to_float(dip.im_read(path2)), Down_Sampled_Matrix)
        Training_labels[i] = 1
        Training_labels[i+50] = 0

elif mode == 1:

    for i in range(50):

        path1 = "Tumor_Images/" + str(i+1) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 1) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X2 = SkullAndShape(cv2.imread(path1))

        Training_images[i, :, :] = dip.resample(X1[0], Down_Sampled_Matrix)
        Training_images[i+50, :, :] = dip.resample(X2[0], Down_Sampled_Matrix)
        Training_labels[i] = 1
        Training_labels[i+50] = 0

elif mode == 2:

    for i in range(50):

        path1 = "Tumor_Images/" + str(i+1) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 1) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X1 = NoiseRemoval(X1[0], 0)
        X2 = SkullAndShape(cv2.imread(path1))
        X2 = NoiseRemoval(X2[0], 0)

        Training_images[i, :, :] = dip.resample(X1, Down_Sampled_Matrix)
        Training_images[i+50, :, :] = dip.resample(X2, Down_Sampled_Matrix)
        Training_labels[i] = 1
        Training_labels[i+50] = 0

elif mode == 3:

    for i in range(50):

        path1 = "Tumor_Images/" + str(i+1) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 1) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X1 = NoiseRemoval(X1[0], 1)
        X2 = SkullAndShape(cv2.imread(path1))
        X2 = NoiseRemoval(X2[0], 1)

        Training_images[i, :, :] = dip.resample(X1, Down_Sampled_Matrix)
        Training_images[i+50, :, :] = dip.resample(X2, Down_Sampled_Matrix)
        Training_labels[i] = 1
        Training_labels[i+50] = 0

elif mode == 4:

    for i in range(50):

        path1 = "Tumor_Images/" + str(i+1) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 1) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X1 = NoiseRemoval(X1[0], 2)
        X2 = SkullAndShape(cv2.imread(path1))
        X2 = NoiseRemoval(X2[0], 2)

        Training_images[i, :, :] = dip.resample(X1, Down_Sampled_Matrix)
        Training_images[i+50, :, :] = dip.resample(X2, Down_Sampled_Matrix)
        Training_labels[i] = 1
        Training_labels[i+50] = 0

elif mode == 5:

    for i in range(50):

        path1 = "Tumor_Images/" + str(i+1) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 1) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X1 = NoiseRemoval(X1, 3)
        X2 = SkullAndShape(cv2.imread(path1))
        X2 = NoiseRemoval(X2, 3)

        Training_images[i, :, :] = dip.resample(X1[0], Down_Sampled_Matrix)
        Training_images[i+50, :, :] = dip.resample(X2[0], Down_Sampled_Matrix)
        Training_labels[i] = 1
        Training_labels[i+50] = 0


# Generate Testing Set
Testing_images = np.zeros([20, 64, 64])
Testing_labels = np.zeros([20, 1])

if mode == 0:

    for i in range(10):
        path1 = "Tumor_Images/" + str(i + 51) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 51) + ".png"
        Testing_images[i, :, :] = dip.resample(dip.im_to_float(dip.im_read(path1)), Down_Sampled_Matrix)
        Testing_images[i+10, :, :] = dip.resample(dip.im_to_float(dip.im_read(path2)), Down_Sampled_Matrix)
        Testing_labels[i] = 1
        Testing_labels[i+10] = 0

elif mode == 1:

    for i in range(10):
        path1 = "Tumor_Images/" + str(i + 51) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 51) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X2 = SkullAndShape(cv2.imread(path1))

        Testing_images[i, :, :] = dip.resample(X1[0], Down_Sampled_Matrix)
        Testing_images[i + 10, :, :] = dip.resample(X2[0], Down_Sampled_Matrix)
        Testing_labels[i] = 1
        Testing_labels[i + 10] = 0

elif mode == 2:

    for i in range(10):
        path1 = "Tumor_Images/" + str(i + 51) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 51) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X1 = NoiseRemoval(X1[0], 0)
        X2 = SkullAndShape(cv2.imread(path1))
        X2 = NoiseRemoval(X2[0], 0)

        Testing_images[i, :, :] = dip.resample(X1, Down_Sampled_Matrix)
        Testing_images[i + 10, :, :] = dip.resample(X2, Down_Sampled_Matrix)
        Testing_labels[i] = 1
        Testing_labels[i + 10] = 0

elif mode == 3:

    for i in range(10):
        path1 = "Tumor_Images/" + str(i + 51) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 51) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X1 = NoiseRemoval(X1[0], 1)
        X2 = SkullAndShape(cv2.imread(path1))
        X2 = NoiseRemoval(X2[0], 1)

        Testing_images[i, :, :] = dip.resample(X1, Down_Sampled_Matrix)
        Testing_images[i + 10, :, :] = dip.resample(X2, Down_Sampled_Matrix)
        Testing_labels[i] = 1
        Testing_labels[i + 10] = 0

elif mode == 4:

    for i in range(10):
        path1 = "Tumor_Images/" + str(i + 51) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 51) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X1 = NoiseRemoval(X1[0], 2)
        X2 = SkullAndShape(cv2.imread(path1))
        X2 = NoiseRemoval(X2[0], 2)

        Testing_images[i, :, :] = dip.resample(X1, Down_Sampled_Matrix)
        Testing_images[i + 10, :, :] = dip.resample(X2, Down_Sampled_Matrix)
        Testing_labels[i] = 1
        Testing_labels[i + 10] = 0

elif mode == 5:

    for i in range(10):
        path1 = "Tumor_Images/" + str(i + 51) + ".png"
        path2 = "Non_Tumor_Images/" + str(i + 51) + ".png"

        X1 = SkullAndShape(cv2.imread(path1))
        X1 = NoiseRemoval(X1, 3)
        X2 = SkullAndShape(cv2.imread(path1))
        X2 = NoiseRemoval(X2, 3)

        Testing_images[i, :, :] = dip.resample(X1[0], Down_Sampled_Matrix)
        Testing_images[i + 10, :, :] = dip.resample(X2[0], Down_Sampled_Matrix)
        Testing_labels[i] = 1
        Testing_labels[i + 10] = 0


## Initialization steps for CNN
img_rows, img_cols = 64, 64
shape_ord = (img_rows, img_cols, 1)
X_train = Training_images.reshape((Training_images.shape[0],) + shape_ord)
X_test = Testing_images.reshape((Testing_images.shape[0],) + shape_ord)

nb_classes = 2
Y_train = np_utils.to_categorical(Training_labels, nb_classes)
Y_test = np_utils.to_categorical(Testing_labels, nb_classes)

nb_epoch = 5
batch_size = 64
nb_filters = 64 # number of convolutional filters to use
nb_pool = 5 # size of pooling area for max pooling
nb_conv = 5 # convolution kernel size
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)


## Construct CNN
model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', input_shape=shape_ord))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
model.evaluate(X_test, Y_test, verbose=0)