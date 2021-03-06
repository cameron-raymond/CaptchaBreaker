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
    print("parsing data")
    # Split the training data into separate train and test sets
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)
    print("encoding data")
    # Save the mapping from labels to one-hot encodings.
    # We'll need this later when we use the model to decode what it's predictions mean
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)
    print('Creating edges')
    # Build the neural network!
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))

    # Output layer with 32 nodes (one for each possible letter/number we predict)
    model.add(Dense(32, activation="softmax"))
    print("compiling model")
    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("fitting data")
    # Train the neural network
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
    print("saving model")
    # Save the trained model to disk
    model.save(MODEL_FILENAME)

if __name__=="__main__":
    # initialize the data and labels
    data = []
    labels = []

    for imageFile in paths.list_images(LETTER_IMAGES_FOLDER):
        img = cv2.imread(imageFile)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = resize_to_fit(img,20,20)
        img = np.expand_dims(img, axis=2)
        
        label = imageFile.split(os.path.sep)[-2]

        data.append(img)
        labels.append(label)
    trainModel(data,labels)








