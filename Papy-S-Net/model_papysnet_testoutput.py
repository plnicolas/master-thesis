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
import keras.applications.nasnet
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

import FragmentSequence as fs
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
    
    #Papy-S-Net (Pirrone et al. 2019)
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(heightImage, widthImage, 3)))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(heightImage, widthImage, 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    
    #Siamese network; two input images
    model1 = model(a)
    model2 = model(b)

    #Use the absolute difference as the similarity measure between the two fragments' feature maps
    sub = keras.layers.Subtract()([model1, model2])
    distance = keras.layers.Lambda(keras.backend.abs)(sub)

    #Two fully connected layers
    fullyConnected1 = keras.layers.Dense(512, activation='relu')(distance)
    fullyConnected2 = keras.layers.Dense(512, activation='relu')(fullyConnected1)

    #Binary classification: same papyrus or not
    lastLayer = keras.layers.Dense(2, activation = 'softmax')(fullyConnected2)

    finalModel = keras.models.Model(inputs=[a,b], outputs=lastLayer)
    
    
    finalModel.compile(optimizer = keras.optimizers.Adam(lr = initialLearningRate), loss = keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    
    finalModel.summary()
    
    return finalModel


def train_network(model, learningSetGenerator, validationSetGenerator, numberEpochs, numberWorkers, batchSize, initialLearningRate, maxQueueSize, numberEpochsLearningRate, discountFactor, prefixResults, stringInformation):
    """
    This method trains a compiled neural network model. The results are recorded for tensorboard and in a csv file. The trained model is also stored. A learning rate scheduler is also used. All the information concerning the model are stored in a model description file.
    
    parameters:
    -----------
    - model: The compiled model to train.
    - learningSetGenerator: The generator of the learning set.
    - validationSetGenerator: The generator of the validation set.
    - numberEpochs: The number of epochs to train.
    - numberWorkers: The number of processes to create.
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
    
    with open("{}/information_model.txt".format(prefixResults + currentTime), mode = "w") as informationFile:
        informationFile.write(stringInformation)
    
    
    tensorboardLogger = keras.callbacks.TensorBoard(log_dir = "{}/tensorboard_log".format(prefixResults + currentTime), histogram_freq = 0, batch_size = batchSize, write_grads = False, write_images = True, update_freq = "epoch")
    
    csvLogger = keras.callbacks.CSVLogger("{}/csv_log.csv".format(prefixResults + currentTime), separator = ",")
    learningRateScheduler = keras.callbacks.LearningRateScheduler(schedule_learning_rate_decorator(initialLearningRate, numberEpochsLearningRate, discountFactor), verbose = 1)

    for i in range(10):
        model.fit_generator(learningSetGenerator, epochs = 1, callbacks = [tensorboardLogger, csvLogger, learningRateScheduler], validation_data = validationSetGenerator, max_queue_size = maxQueueSize, workers = numberWorkers, use_multiprocessing = True, verbose = 2)
        #Evaluate the model on test set and compute metrics
        y_pred = model.predict_generator(validationSetGenerator, max_queue_size = maxQueueSize, workers = numberWorkers, use_multiprocessing = True)
        y_pred_bool = numpy.argmax(y_pred, axis=1)

        print(y_pred[:100].tolist())

        print(classification_report(y_test, y_pred_bool))
    
    #model.save("{}/model_trained.h5".format(prefixResults + currentTime))
    
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
    This class stores different parameters used in the code.
    """
    
    def __init__(self, sizeBatch, numberEpochs, initialLearningRate, numberEpochsLearningRate, discountFactor, widthImage, heightImage, numberWorkers, maxQueueSize, pathImages, prefixResults, additionalInformation):
        """
        This is the initialization method.
        
        typeIdentification: The type of identification ("artist", "genre", or "style").
        sizeBatch: The size of the batch.
        numberEpochs: The number of epochs to train.
        initialLearningRate: The initial learning rate.
        numberEpochsLearningRate: The number of epochs between two changes of the learning rate.
        discountFactor: The discount factor.
        widthImage: The width of the images.
        heightImage: The height of the images.
        numberWorkers: The number of processes to create for parallel computing.
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
        self.numberWorkers = numberWorkers
        self.maxQueueSize = maxQueueSize
        self.pathImages = pathImages
        self.prefixResults = prefixResults
        self.additionalInformation = additionalInformation
    
    
    def __str__(self):
        """
        This method returns a string representing the object.
        """
        
        stringInformation = "SIZE_BATCH: {}\nNUMBER_EPOCHS: {}\nINITIAL_LEARNING_RATE: {}\nNUMBER_EPOCHS_LEARNING_RATE: {}\nDISCOUNT_FACTOR: {}\nWIDTH_IMAGE: {}\nHEIGHT_IMAGE: {}\nPROBABILITY_HORIZONTAL_FLIP: {}\nPROBABILITY_VERTICAL_FLIP: {}\nPROBABILITY_CROP_LEARNING_SET: {}\nREDUCTION_OPERATION_TEST_SET: {}\nNUMBER_WORKERS: {}\nMAX_QUEUE_SIZE: {}\nPATH_IMAGES: {}\nPREFIX_RESULTS: {}\n\nADDITIONAL_INFORMATION:\n{}".format(self.sizeBatch, self.numberEpochs, self.initialLearningRate, self.numberEpochsLearningRate, self.discountFactor, self.widthImage, self.heightImage, self.probabilityHorizontalFlip, self.probabilityVerticalFlip, self.probabilityCropLearningSet, self.reductionOperationTestSet, self.numberWorkers, self.maxQueueSize, self.pathLists, self.pathImages, self.prefixResults, self.additionalInformation)
        
        return stringInformation


def sample_pairs(K, data, IDList):
    pairs = []
    labels = []

    #For each papyrus used for training
    for index in IDList:
        isIndex = data.iloc[:,1]==index
        isNotIndex = data.iloc[:,1]!=index
        #List of images from the indexed papyrus
        indexTrueList = data[isIndex].iloc[:,0]
        #List of images NOT from the indexed papyrus
        indexFalseList = data[isNotIndex].iloc[:,0]
           
        #K negative pairs
        p1List = indexTrueList.sample(n=K, replace=True, random_state=356)
        p2List = indexFalseList.sample(n=K, replace=True, random_state=323)
        #pairs.append(pandas.concat([p1.reset_index(drop=True), p2.reset_index(drop=True)], axis=1).values)
        for k in range(K):
            pair = [p1List.values[k], p2List.values[k]]
            if pair not in pairs:
                pairs.append(pair)
                labels.append(1)
        
        #K positive pairs
        p1List = indexTrueList.sample(n=K, replace=True, random_state=362)
        p2List = indexTrueList.sample(n=K, replace=True, random_state=316)
        #pairs.append(pandas.concat([p1.reset_index(drop=True), p2.reset_index(drop=True)], axis=1).values)
        for k in range(K):
            pair = [p1List.values[k], p2List.values[k]]
            if pair not in pairs:
                pairs.append(pair)
                labels.append(0)

    #Shuffle the pairs and label lists before returning them
    #The two lists are shuffled at once with the same order, of course
    tmp = list(zip(pairs, labels))
    random.shuffle(tmp)
    pairs, labels = zip(*tmp)

    return pairs, labels

def create_pairs(K, data, IDList, IDRange):
    """
    Function to create fragment pairs for training and testing.
    In order to avoid any bias, the papyri used to generate the training pairs are NOT used to generate the testing pairs.

    Parameters:
    ----------
        - K: The number of pairs of each type (positive and negative) to sample. Duplicates will be dropped,
        so the final number of pairs WILL be smaller than 2K
        - data: CSV file generated using retrieve_annotations_crop.py or retrieve_annotations_alphamask.py
        - IDList: The list of IDs of the different papyri
        - IDRange: Defines the range of papyri to use to generate the training pairs. The rest of the papyri
        will be used to generate the testing pairs.

    Returns:
    --------
        - The training pairs and their associated labels, the testing pairs and their associated labels.

    """
    pairsTrain, labelsTrain = sample_pairs(K, data, IDList[:IDRange])   
    pairsTest, labelsTest = sample_pairs(K, data, IDList[IDRange:])

    return pairsTrain, labelsTrain, pairsTest, labelsTest


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
    NUMBER_EPOCHS = 10
    INITIAL_LEARNING_RATE = 0.0001
    NUMBER_EPOCHS_LEARNING_RATE = 20
    DISCOUNT_FACTOR = 1
    WIDTH_IMAGE = 128
    HEIGHT_IMAGE = 128
    PROBABILITY_HORIZONTAL_FLIP = 0.5
    PROBABILITY_VERTICAL_FLIP = 0.5
    NUMBER_WORKERS = multiprocessing.cpu_count()
    MAX_QUEUE_SIZE = 50
    #PATH_IMAGES = ""
    PATH_IMAGES = "/scratch/plnicolas/datasets/"
    #PATH_CSV = "dataset.csv"
    PATH_CSV = "/home/plnicolas/codes/dataset.csv"
    #PREFIX_RESULTS = "Results/"
    PREFIX_RESULTS = "/home/plnicolas/codes/Results/Papy-S-Net/"
    ADDITIONAL_INFORMATION = "This model implements a siamese neural network using Papy-S-Net (Pirrone '2019) trained from scratch. The similarity measure is the absolute difference and the last layer is a dense layer with a softmax activation function. All weights are directly trainable. The loss function is the categorical cross-entropy. The optimizer is Adam with the default beta1 and beta2 parameters."
    
    stringInformation = "PAIRS: {}\nSIZE_BATCH: {}\nNUMBER_EPOCHS: {}\nINITIAL_LEARNING_RATE: {}\nNUMBER_EPOCHS_LEARNING_RATE: {}\nDISCOUNT_FACTOR: {}\nWIDTH_IMAGE: {}\nHEIGHT_IMAGE: {}\nNUMBER_WORKERS: {}\nMAX_QUEUE_SIZE: {}\nPATH_IMAGES: {}\nPREFIX_RESULTS: {}\n\nADDITIONAL_INFORMATION:\n{}".format(PAIRS, SIZE_BATCH, NUMBER_EPOCHS, INITIAL_LEARNING_RATE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, WIDTH_IMAGE, HEIGHT_IMAGE, NUMBER_WORKERS, MAX_QUEUE_SIZE, PATH_IMAGES, PREFIX_RESULTS, ADDITIONAL_INFORMATION)
    
    data = pandas.read_csv(PATH_CSV, sep=",", header=None)

    #IDList constains the ID of each papyrus
    IDList = data.iloc[:,1].drop_duplicates().values

    #IDRange is the proportion of papyri we will use to generate the training pairs
    #The remaining papyri will be used to generate the testing pairs => unbiased
    IDRange = (int) (len(IDList) - (len(IDList)/4))
    print("IDRange: {}".format(IDRange))

    X_train, y_train, X_test, y_test = create_pairs(PAIRS, data, IDList, IDRange)

    print("Number of training pairs: {}".format(len(X_train)))
    print("Number of testing pairs: {}".format(len(X_test)))

    model = create_neural_network(WIDTH_IMAGE, HEIGHT_IMAGE, INITIAL_LEARNING_RATE)
    
    learningSequence = fs.FragmentSequence(X_train, y_train, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES, PROBABILITY_HORIZONTAL_FLIP, PROBABILITY_VERTICAL_FLIP)
    validationSequence = fsv.FragmentSequenceValidation(X_test, y_test, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)
    
    currentTime = train_network(model, learningSequence, validationSequence, NUMBER_EPOCHS, NUMBER_WORKERS, SIZE_BATCH, INITIAL_LEARNING_RATE, MAX_QUEUE_SIZE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, PREFIX_RESULTS, stringInformation)
        
    parametersClass = ParametersClass(SIZE_BATCH, NUMBER_EPOCHS, INITIAL_LEARNING_RATE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, WIDTH_IMAGE, HEIGHT_IMAGE, NUMBER_WORKERS, MAX_QUEUE_SIZE, PATH_IMAGES, PREFIX_RESULTS, ADDITIONAL_INFORMATION)