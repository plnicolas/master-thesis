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
import keras.applications.resnet50
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

import EvaluationPipeline
import PairGenerator

import FragmentSequence as fs
import FragmentSequenceValidation as fsv

from keras.models import Model
# This function is needed to be able to save/load the model
# (weird workaround for a TF bug...)
def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

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

    # ResNet50
    model.add(keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(heightImage, widthImage, 3), pooling='avg'))
    
    # Siamese network; two input images
    model1 = model(a)
    model2 = model(b)

    # Use the absolute difference as the similarity measure between the two fragments' feature maps
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
    
    pathResults = prefixResults + currentTime

    if not os.path.exists(pathResults):
        os.makedirs(pathResults)
    
    with open("{}/information_model.txt".format(pathResults), mode="w") as informationFile:
        informationFile.write(stringInformation)
      
    csvLogger = keras.callbacks.CSVLogger("{}/csv_log.csv".format(pathResults), separator=",")
    learningRateScheduler = keras.callbacks.LearningRateScheduler(schedule_learning_rate_decorator(initialLearningRate, numberEpochsLearningRate, discountFactor), verbose = 1)
    
    model.fit_generator(learningSetGenerator, epochs=numberEpochs, callbacks=[csvLogger, learningRateScheduler], validation_data=validationSetGenerator, max_queue_size=maxQueueSize, workers=multiprocessing.cpu_count(), use_multiprocessing=True, verbose=2)
    
    # To save the model
    modelFreezed = freeze_layers(model)
    modelFreezed.save("{}/model_trained.h5".format(pathResults))

    return currentTime


def schedule_learning_rate_decorator(initialLR, numberEpochsLR, discountFactor):
    """
    This function returns the learning rate scheduler.
    
    Parameters:
    -----------
    - initialLR: The initial learning rate.
    - numberEpochsLR: The number of epochs between two changes of the learning rate.
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
        
        return initialLR * (discountFactor ** (epochIndex // numberEpochsLR))
        
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

    def __init__(self, sizeBatch, numberEpochs, initialLearningRate, numberEpochsLearningRate, discountFactor, widthImage, heightImage, maxQueueSize, pathImages, prefixResults, additionalInformation):
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
        self.pathImages = pathImages
        self.prefixResults = prefixResults
        self.additionalInformation = additionalInformation

    def __str__(self):
        """
        This method returns a string representing the object.
        """

        stringInformation = "SIZE_BATCH: {}\nNUMBER_EPOCHS: {}\nINITIAL_LEARNING_RATE: {}\nNUMBER_EPOCHS_LEARNING_RATE: {}\nDISCOUNT_FACTOR: {}\nWIDTH_IMAGE: {}\nHEIGHT_IMAGE: {}\nPROBABILITY_HORIZONTAL_FLIP: {}\nPROBABILITY_VERTICAL_FLIP: {}\nPROBABILITY_CROP_LEARNING_SET: {}\nREDUCTION_OPERATION_TEST_SET: {}\nMAX_QUEUE_SIZE: {}\nPATH_IMAGES: {}\nPREFIX_RESULTS: {}\n\nADDITIONAL_INFORMATION:\n{}".format(
            self.sizeBatch, self.numberEpochs, self.initialLearningRate, self.numberEpochsLearningRate, self.discountFactor, self.widthImage, self.heightImage, self.probabilityHorizontalFlip, self.probabilityVerticalFlip, self.probabilityCropLearningSet, self.reductionOperationTestSet, self.maxQueueSize, self.pathLists, self.pathImages, self.prefixResults, self.additionalInformation)

        return stringInformation


if __name__ == "__main__":
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
    SIZE_BATCH = 16
    NUMBER_EPOCHS = 40
    INITIAL_LEARNING_RATE = 0.0001
    NUMBER_EPOCHS_LEARNING_RATE = 20
    DISCOUNT_FACTOR = 0.1
    WIDTH_IMAGE = 224
    HEIGHT_IMAGE = 224
    PROBABILITY_HORIZONTAL_FLIP = 0.5
    PROBABILITY_VERTICAL_FLIP = 0.5
    NUMBER_WORKERS = multiprocessing.cpu_count()
    MAX_QUEUE_SIZE = 50

    PATH_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
    #PATH_IMAGES = "/scratch/users/plnicolas/datasets/"
    PATH_CSV = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset.csv"
    #PATH_CSV = "/home/plnicolas/codes/dataset.csv"
    PREFIX_RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Results/ResNet50/IN/"
    #PREFIX_RESULTS = "/home/plnicolas/codes/Results/ResNet50/IN/"
    ADDITIONAL_INFORMATION = "ResNet50-Twin with ImageNet weights. All weights are directly trainable. The loss function is the categorical cross-entropy. The optimizer is Adam with the default beta1 and beta2 parameters."
    
    stringInformation = "PAIRS: {}\nSIZE_BATCH: {}\nNUMBER_EPOCHS: {}\nINITIAL_LEARNING_RATE: {}\nNUMBER_EPOCHS_LEARNING_RATE: {}\nDISCOUNT_FACTOR: {}\nWIDTH_IMAGE: {}\nHEIGHT_IMAGE: {}\nNUMBER_WORKERS: {}\nMAX_QUEUE_SIZE: {}\nPATH_IMAGES: {}\nPREFIX_RESULTS: {}\n\nADDITIONAL_INFORMATION:\n{}".format(PAIRS, SIZE_BATCH, NUMBER_EPOCHS, INITIAL_LEARNING_RATE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, WIDTH_IMAGE, HEIGHT_IMAGE, NUMBER_WORKERS, MAX_QUEUE_SIZE, PATH_IMAGES, PREFIX_RESULTS, ADDITIONAL_INFORMATION)
    
    # Generate the training and test pairs
    X_train, y_train, X_test, y_test = PairGenerator.create_pairs(PAIRS, PATH_CSV)

    print("Number of training pairs: {}".format(len(X_train)))
    print("Number of testing pairs: {}".format(len(X_test)))

    model = create_neural_network(WIDTH_IMAGE, HEIGHT_IMAGE, INITIAL_LEARNING_RATE)

    learningSequence = fs.FragmentSequence(X_train, y_train, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES, PROBABILITY_HORIZONTAL_FLIP, PROBABILITY_VERTICAL_FLIP)
    validationSequence = fsv.FragmentSequenceValidation(X_test, y_test, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)

    currentTime = train_network(model, learningSequence, validationSequence, NUMBER_EPOCHS, SIZE_BATCH, INITIAL_LEARNING_RATE, MAX_QUEUE_SIZE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, PREFIX_RESULTS, stringInformation)

    parametersClass = ParametersClass(SIZE_BATCH, NUMBER_EPOCHS, INITIAL_LEARNING_RATE, NUMBER_EPOCHS_LEARNING_RATE,
                                      DISCOUNT_FACTOR, WIDTH_IMAGE, HEIGHT_IMAGE, MAX_QUEUE_SIZE, PATH_IMAGES, PREFIX_RESULTS, ADDITIONAL_INFORMATION)

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
    EvaluationPipeline.run_pipeline(model, PATH_CSV, pathResults)

    with open("{}/information_model_binary.pkl".format(pathResults), "wb") as f:
        pickle.dump(parametersClass, f)