import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras import backend as K
import keras.utils
import keras.models
import keras.layers
import keras.callbacks
import keras.metrics
import keras.preprocessing.image
import keras.optimizers
import keras.losses
import datetime
import skimage.io
import skimage.transform
import multiprocessing
import numpy.random
import os
import random
import pickle

from argparse import ArgumentParser

import EvaluationPipeline
import PairGenerator

import FragmentSequence as fs
import FragmentSequenceBright as fsb
import FragmentSequenceValidation as fsv

def create_neural_network(widthImage, heightImage, initialLearningRate):
    """
    This function creates a compiled neural network model and displays a summary of it.

    parameters:
    -----------
    - widthImage: The width of the images.
    - heightImage: The height of the images.
    - initialLearningRate: The initial learning rate.

    returns:
    --------
    - model: The compiled neural network model created.
    """
    a = keras.layers.Input((heightImage, widthImage, 3))
    b = keras.layers.Input((heightImage, widthImage, 3))

    model = keras.models.Sequential()

    # Papy-S-Net (Pirrone et al. 2019)
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(
        1, 1), activation='relu', input_shape=(heightImage, widthImage, 3)))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(
        1, 1), activation='relu', input_shape=(heightImage, widthImage, 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())

    # Siamese network; two input images
    model1 = model(a)
    model2 = model(b)

    # Use the absolute difference as the similarity measure between the two
    # fragments' feature maps
    sub = keras.layers.Subtract()([model1, model2])
    distance = keras.layers.Lambda(keras.backend.abs)(sub)

    # Two fully connected layers
    fullyConnected1 = keras.layers.Dense(512, activation='relu')(distance)
    fullyConnected2 = keras.layers.Dense(512, activation='relu')(fullyConnected1)

    # Binary classification: same papyrus or not
    lastLayer = keras.layers.Dense(2, activation='softmax')(fullyConnected2)

    finalModel = keras.models.Model(inputs=[a,b], outputs=lastLayer)

    finalModel.compile(optimizer=keras.optimizers.Adam(lr=initialLearningRate), loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    finalModel.summary()

    return finalModel


def train_network(model, learningSetGenerator, validationSetGenerator, numberEpochs, batchSize, initialLearningRate, maxQueueSize, numberEpochsLearningRate, discountFactor, prefixResults, stringInformation):
    """
    This method trains a compiled neural network model. The results are recorded for tensorboard and in a csv file. The trained model is also stored. A learning rate scheduler is also used. All the information concerning the model are stored in a model description file.

    parameters:
    -----------
    - model: The compiled model to train.
    - learningSetGenerator: The generator of the learning set.
    - validationSetGenerator: The generator of the validation set.
    - numberEpochs: The number of epochs to train.
    - batchSize: The size of each batch.
    - initialLearningRate: The initial learning rate.
    - maxQueueSize: The maximum size of the queue of preprocessed batches.
    - numberEpochsLearningRate: The number of epochs between two changes of the learning rate.
    - discountFactor: The discount factor to use for changing the learning rate.
    - prefixResults: The prefix of the path where to store the results.
    - stringInformation: The additional information to write to the model description file.
    """

    currentTime = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    if not os.path.exists(prefixResults + currentTime):
        os.makedirs(prefixResults + currentTime)

    with open("{}/information_model.txt".format(prefixResults + currentTime), mode="w") as informationFile:
        informationFile.write(stringInformation)

    csvLogger = keras.callbacks.CSVLogger("{}/csv_log.csv".format(prefixResults + currentTime), separator=",")
    learningRateScheduler = keras.callbacks.LearningRateScheduler(schedule_learning_rate_decorator(initialLearningRate, numberEpochsLearningRate, discountFactor), verbose=1)

    model.fit_generator(learningSetGenerator, epochs=numberEpochs, callbacks=[csvLogger, learningRateScheduler], validation_data=validationSetGenerator, max_queue_size=maxQueueSize, workers=multiprocessing.cpu_count(), use_multiprocessing=True, verbose=2)

    # To save the model
    model.save("{}/model_trained.h5".format(prefixResults + currentTime))

    return currentTime


def schedule_learning_rate_decorator(initialLearningRate, numberEpochsLearningRate, discountFactor):
    """
    This function returns the learning rate scheduler.

    Parameters:
    -----------
    - initialLearningRate: The initial learning rate.
    - numberEpochsLearningRate: The number of epochs between two changes of the learning rate.
    - discountFactor: The discount factor.

    Returns:
    --------
    - The learning rate scheduler.
    """

    def schedule_learning_rate(epochIndex):
        """
        This function defines the learning rate scheduler.

        Parameters:
        ----------
        - epochIndex: The index of the epoch that is going to start.

        Returns:
        --------
        - The new learning rate.
        """

        return initialLearningRate * (discountFactor ** (epochIndex // numberEpochsLearningRate))

    return schedule_learning_rate


class ArgumentException(Exception):
    """
    This class defines an exception on an argument of a function.
    """

    def __init__(self, message):
        """
        This is the initialization function.

        parameter:
        ----------
        - message: The message to provide.
        """

        self.message = message


class ParametersClass:
    """
    This class stores the different parameters used to train the network.
    """

    def __init__(self, sizeBatch, numberEpochs, initialLearningRate, numberEpochsLearningRate, discountFactor, widthImage, heightImage, maxQueueSize, brightnessShifts, pathImages, prefixResults, additionalInformation):
        """
        This is the initialization method.

        sizeBatch: The batch size.
        numberEpochs: The number of epochs to train.
        initialLearningRate: The initial learning rate.
        numberEpochsLearningRate: The number of epochs between two changes of the learning rate.
        discountFactor: The discount factor.
        widthImage: The width of the images.
        heightImage: The height of the images.
        maxQueueSize: The maximum size of the queue of preprocessed batches.
        pathImages: The path to give as prefix for every path stored in the dataset of paths towards images.
        prefixResults: The prefix of the path where to store the results.
        additionalInformation: The additional information to write to the model description file.
        """

        self.sizeBatch = sizeBatch
        self.numberEpochs = numberEpochs
        self.initialLearningRate = initialLearningRate
        self.numberEpochsLearningRate = numberEpochsLearningRate
        self.discountFactor = discountFactor
        self.widthImage = widthImage
        self.heightImage = heightImage
        self.maxQueueSize = maxQueueSize
        self.brightnessShifts = brightnessShifts
        self.pathImages = pathImages
        self.prefixResults = prefixResults
        self.additionalInformation = additionalInformation

    def __str__(self):
        """
        This method returns a string representing the object.
        """

        stringInformation = "SIZE_BATCH: {}\nNUMBER_EPOCHS: {}\nINITIAL_LEARNING_RATE: {}\nNUMBER_EPOCHS_LEARNING_RATE: {}\nDISCOUNT_FACTOR: {}\nWIDTH_IMAGE: {}\nHEIGHT_IMAGE: {}\nPROBABILITY_HORIZONTAL_FLIP: {}\nPROBABILITY_VERTICAL_FLIP: {}\nPROBABILITY_CROP_LEARNING_SET: {}\nREDUCTION_OPERATION_TEST_SET: {}\nMAX_QUEUE_SIZE: {}\nBRIGHTNESS_SHIFTS: {}\nPATH_IMAGES: {}\nPREFIX_RESULTS: {}\n\nADDITIONAL_INFORMATION:\n{}".format(
            self.sizeBatch, self.numberEpochs, self.initialLearningRate, self.numberEpochsLearningRate, self.discountFactor, self.widthImage, self.heightImage, self.probabilityHorizontalFlip, self.probabilityVerticalFlip, self.probabilityCropLearningSet, self.reductionOperationTestSet, self.maxQueueSize, self.brightnessShifts, self.pathLists, self.pathImages, self.prefixResults, self.additionalInformation)

        return stringInformation


def get_arguments():
    # Get the arguments of the program
    parser = ArgumentParser(prog="Papy-S-Net architecture")

    parser.add_argument('--size', dest='size', default=128, type=int, help="Image size (square; only one value needed)")
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--brightness', dest='brightness', default=0, type=int, help="Brightness shifts during training (0 = no)")

    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_arguments()

    """
    PAIRS: The number of pairs of each type (positive/negative) to sample for each papyrus; duplicates will be discarded.
    SIZE_BATCH: The size of the batch.
    NUMBER_EPOCHS: The number of epochs to train.
    INITIAL_LEARNING_RATE: The initial learning rate.
    NUMBER_EPOCHS_LEARNING_RATE: The number of epochs between two changes of the learning rate.
    DISCOUNT_FACTOR: The discount factor.
    WIDTH_IMAGE: The width of the images.
    HEIGHT_IMAGE: The height of the images.
    NUMBER_WORKERS: The number of processes to create for parallel computing.
    MAX_QUEUE_SIZE: The maximum size of the queue of preprocessed batches.
    PATH_IMAGES: The path to give as prefix for every path stored in the dataset of paths towards images.
    PREFIX_RESULTS: The prefix of the path where to store the results.
    ADDITIONAL_INFORMATION: The additional information to write to the model description file.
    """
    PAIRS = 4000
    SIZE_BATCH = args.batch_size
    NUMBER_EPOCHS = 40
    INITIAL_LEARNING_RATE = 0.00005
    NUMBER_EPOCHS_LEARNING_RATE = 20
    DISCOUNT_FACTOR = 0.1
    WIDTH_IMAGE = args.size
    HEIGHT_IMAGE = args.size
    PROBABILITY_HORIZONTAL_FLIP = 0.5
    PROBABILITY_VERTICAL_FLIP = 0.5
    NUMBER_WORKERS = multiprocessing.cpu_count()
    MAX_QUEUE_SIZE = 50
    BRIGHTNESS_SHIFTS = args.brightness
    PATH_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
    #PATH_IMAGES = "/scratch/users/plnicolas/datasets/"
    PATH_CSV = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset.csv"
    #PATH_CSV = "/home/plnicolas/codes/dataset.csv"
    PREFIX_RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Results/Papy-S-Net/Crop/"
    #PREFIX_RESULTS = "/home/plnicolas/codes/Results/Papy-S-Net/Crop/"
    ADDITIONAL_INFORMATION = "This model implements a siamese neural network using Papy-S-Net (Pirrone '2019) trained from scratch. The similarity measure is the absolute difference and the last layer is a dense layer with a softmax activation function. All weights are directly trainable. The loss function is the categorical cross-entropy. The optimizer is Adam with the default beta1 and beta2 parameters."

    stringInformation = "PAIRS: {}\nSIZE_BATCH: {}\nNUMBER_EPOCHS: {}\nINITIAL_LEARNING_RATE: {}\nNUMBER_EPOCHS_LEARNING_RATE: {}\nDISCOUNT_FACTOR: {}\nWIDTH_IMAGE: {}\nHEIGHT_IMAGE: {}\nMAX_QUEUE_SIZE: {}\nBRIGHTNESS_SHIFTS: {}\nPATH_IMAGES: {}\nPREFIX_RESULTS: {}\n\nADDITIONAL_INFORMATION:\n{}".format(
        PAIRS, SIZE_BATCH, NUMBER_EPOCHS, INITIAL_LEARNING_RATE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, WIDTH_IMAGE, HEIGHT_IMAGE, MAX_QUEUE_SIZE, BRIGHTNESS_SHIFTS, PATH_IMAGES, PREFIX_RESULTS, ADDITIONAL_INFORMATION)

    # Generate the training and test pairs
    X_train, y_train, X_test, y_test = PairGenerator.create_pairs(PAIRS, PATH_CSV)

    print("Number of training pairs: {}".format(len(X_train)))
    print("Number of testing pairs: {}".format(len(X_test)))

    model = create_neural_network(WIDTH_IMAGE, HEIGHT_IMAGE, INITIAL_LEARNING_RATE)

    if BRIGHTNESS_SHIFTS == 0:
        learningSequence = fs.FragmentSequence(X_train, y_train, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES, PROBABILITY_HORIZONTAL_FLIP, PROBABILITY_VERTICAL_FLIP)
    else:
        learningSequence = fsb.FragmentSequenceBright(X_train, y_train, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES, PROBABILITY_HORIZONTAL_FLIP, PROBABILITY_VERTICAL_FLIP)

    validationSequence = fsv.FragmentSequenceValidation(X_test, y_test, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)

    currentTime = train_network(model, learningSequence, validationSequence, NUMBER_EPOCHS, SIZE_BATCH, INITIAL_LEARNING_RATE,
                                MAX_QUEUE_SIZE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, PREFIX_RESULTS, stringInformation)

    parametersClass = ParametersClass(SIZE_BATCH, NUMBER_EPOCHS, INITIAL_LEARNING_RATE, NUMBER_EPOCHS_LEARNING_RATE,
                                      DISCOUNT_FACTOR, WIDTH_IMAGE, HEIGHT_IMAGE, MAX_QUEUE_SIZE, BRIGHTNESS_SHIFTS, PATH_IMAGES, PREFIX_RESULTS, ADDITIONAL_INFORMATION)

    # Evaluate the model on test set and compute global metrics
    y_pred = model.predict_generator(validationSequence, max_queue_size=MAX_QUEUE_SIZE,
                                     workers=multiprocessing.cpu_count(), use_multiprocessing=True)

    y_pred_bool = numpy.argmax(y_pred, axis=1)

    print("Global classification report:")
    print(classification_report(y_test, y_pred_bool))
    print("")

    # Run evaluation pipeline
    print("Running evaluation pipeline...")
    pathResults = PREFIX_RESULTS + currentTime + "/"
    EvaluationPipeline.run_pipeline(model, PATH_CSV, args, pathResults)

    with open("{}/information_model_binary.pkl".format(pathResults), "wb") as f:
        pickle.dump(parametersClass, f)
