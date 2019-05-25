import cv2
import pickle
import os.path
import numpy as np
from imutils import paths,resize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense



LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

def trainModel(data,labels):
    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Split the training data into separate train and test sets
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)


if __name__=="__main":
    # initialize the data and labels
    data = []
    labels = []

    for imageFile in paths.list_images(LETTER_IMAGES_FOLDER):
        img = cv2.imread(imageFile)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = resize_to_fit(img,20,20)
        img = np.expand_dims(img, axis=2)
        
        label = imageFile.split(os.path.sep)[-2]
        print(imageFile,label)

        data.append(image)
        labels.append(label)
    trainModel(data,labels)








