import numpy as np
import pandas
from sklearn.model_selection import train_test_split
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
import pickle

import FragmentSequence as fs
import FragmentSequenceCentered as fsc


def read_database(path):
    """
    This function reads a dataset of paths towards images and the corresponding classes.
    
    parameter:
    ----------
    - path: The path to the dataset file.
    
    returns:
    --------
    - database[0]: The dataset of paths towards images.
    - np.array(database[1]): The corresponding classes.
    """
    
    database = pandas.read_csv(path, sep = ",", header = None)
    
    return database[0], np.array(database[1])
    
    

def read_list(path):
    """
    This function reads the list of classes and the corresponding indexes.
    
    parameter:
    ----------
    - path: The path to the list file.
    
    returns:
    --------
    - np.array(database[0]): The indexes of the classes.
    - database[1]: The corresponding class names.
    """
    
    database = pandas.read_csv(path, delim_whitespace = True, header = None)
    
    return np.array(database[0]), database[1]


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_neural_network(numberClasses, widthImage, heightImage, initialLearningRate):
    """
    This function creates a compiled neural network model and displays a summary of it.
    
    parameters:
    -----------
    - numberClasses: The total number of classes.
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
    
    model.add(keras.applications.resnet50.ResNet50(include_top = False, weights = 'imagenet', input_shape = (heightImage, widthImage, 3), pooling = 'avg'))
    
    model1 = model(a)
    model2 = model(b)

    distance = keras.layers.Lambda(euclidean_distance, output_shape = eucl_dist_output_shape)([model1, model2])

    #Binary classification: same papyrus or not
    lastLayer = keras.layers.Dense(2, activation = 'softmax')(distance)

    finalModel = keras.models.Model(inputs=[a,b], outputs=lastLayer)
    
    
    finalModel.compile(optimizer = keras.optimizers.Adam(lr = initialLearningRate), loss = contrastive_loss, metrics=['accuracy'])
    
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

    
    model.fit_generator(learningSetGenerator, epochs = numberEpochs, callbacks = [tensorboardLogger, csvLogger, learningRateScheduler], validation_data = validationSetGenerator, max_queue_size = maxQueueSize, workers = numberWorkers, use_multiprocessing = True)
    
    
    model.save("{}/model_trained.h5".format(prefixResults + currentTime))
    
    return currentTime


def schedule_learning_rate_decorator(initialLearningRate, numberEpochsLearningRate, discountFactor):
    """
    This function returns the learning rate scheduler.
    
    parameters:
    -----------
    - initialLearningRate: The initial learning rate.
    - numberEpochsLearningRate: The number of epochs between two changes of the learning rate.
    - discountFactor: The discount factor.
    
    returns:
    --------
    - The learning rate scheduler.
    """
    
    def schedule_learning_rate(epochIndex):
        """
        This function defines the learning rate scheduler.
        
        parameter:
        ----------
        - epochIndex: The index of the epoch that is going to start.
        
        returns:
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
    
    def __init__(self, sizeBatch, numberEpochs, initialLearningRate, numberEpochsLearningRate, discountFactor, widthImage, heightImage, probabilityHorizontalFlip, probabilityVerticalFlip, probabilityCropLearningSet, reductionOperationTestSet, numberWorkers, maxQueueSize, pathLists, pathImages, prefixResults, additionalInformation):
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
        probabilityHorizontalFlip: The probability to flip horizontally an image.
        probabilityVerticalFlip: The probability to flip vertically an image. The probability to neither flip horizontally nor vertically is given by 1 - PROBABILITY_HORIZONTAL_FLIP - PROBABILITY_VERTICAL_FLIP
        probabilityCropLearningSet: The probability to use a crop as reduction operation in an imae of the learning set. The probability to resize the image instead of taking a crop is 1 - PROBABILITY_CROP_LEARNING_SET
        reductionOperationTestSet: The reduction operation to use in the test set ("crop" or "resize").
        numberWorkers: The number of processes to create for parallel computing.
        maxQueueSize: The maximum size of the queue of preprocessed batches.
        pathLists: The path to the 3 lists of classes indexes and their corresponding names for each type of identification task.
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
        self.probabilityHorizontalFlip = probabilityHorizontalFlip
        self.probabilityVerticalFlip = probabilityVerticalFlip
        self.probabilityCropLearningSet = probabilityCropLearningSet
        self.reductionOperationTestSet = reductionOperationTestSet
        self.numberWorkers = numberWorkers
        self.maxQueueSize = maxQueueSize
        self.pathLists = pathLists
        self.pathImages = pathImages
        self.prefixResults = prefixResults
        self.additionalInformation = additionalInformation
    
    
    def __str__(self):
        """
        This method returns a string representing the object.
        """
        
        stringInformation = "SIZE_BATCH: {}\nNUMBER_EPOCHS: {}\nINITIAL_LEARNING_RATE: {}\nNUMBER_EPOCHS_LEARNING_RATE: {}\nDISCOUNT_FACTOR: {}\nWIDTH_IMAGE: {}\nHEIGHT_IMAGE: {}\nPROBABILITY_HORIZONTAL_FLIP: {}\nPROBABILITY_VERTICAL_FLIP: {}\nPROBABILITY_CROP_LEARNING_SET: {}\nREDUCTION_OPERATION_TEST_SET: {}\nNUMBER_WORKERS: {}\nMAX_QUEUE_SIZE: {}\nPATH_LISTS: {}\nPATH_IMAGES: {}\nPREFIX_RESULTS: {}\n\nADDITIONAL_INFORMATION:\n{}".format(self.sizeBatch, self.numberEpochs, self.initialLearningRate, self.numberEpochsLearningRate, self.discountFactor, self.widthImage, self.heightImage, self.probabilityHorizontalFlip, self.probabilityVerticalFlip, self.probabilityCropLearningSet, self.reductionOperationTestSet, self.numberWorkers, self.maxQueueSize, self.pathLists, self.pathImages, self.prefixResults, self.additionalInformation)
        
        return stringInformation


def create_pairs(data, indexList):
    #Number of pairs of each type to generate, for each papyrus
    K = 100
    pairs = []
    labels = []
    for index in indexList:
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
            pairs.append([p1List.values[k], p2List.values[k]])
            labels.append(0)
        
        #K positive pairs
        p1List = indexTrueList.sample(n=K, replace=True, random_state=362)
        p2List = indexTrueList.sample(n=K, replace=True, random_state=316)
        #pairs.append(pandas.concat([p1.reset_index(drop=True), p2.reset_index(drop=True)], axis=1).values)
        for k in range(K):
            pairs.append([p1List.values[k], p2List.values[k]])
            labels.append(1)

    return pairs, labels



    return np.array(pairs), np.array(labels)

if __name__ == "__main__":
    """
    TYPE_IDENTIFICATION: The type of identification ("artist", "genre", or "style").
    SIZE_BATCH: The size of the batch.
    NUMBER_EPOCHS: The number of epochs to train.
    INITIAL_LEARNING_RATE: The initial learning rate.
    NUMBER_EPOCHS_LEARNING_RATE: The number of epochs between two changes of the learning rate.
    DISCOUNT_FACTOR: The discount factor.
    WIDTH_IMAGE: The width of the images.
    HEIGHT_IMAGE: The height of the images.
    PROBABILITY_HORIZONTAL_FLIP: The probability to flip horizontally an image.
    PROBABILITY_VERTICAL_FLIP: The probability to flip vertically an image. The probability to neither flip horizontally nor vertically is given by 1 - PROBABILITY_HORIZONTAL_FLIP - PROBABILITY_VERTICAL_FLIP
    PROBABILITY_CROP_LEARNING_SET: The probability to use a crop as reduction operation in an imae of the learning set. The probability to resize the image instead of taking a crop is 1 - PROBABILITY_CROP_LEARNING_SET
    REDUCTION_OPERATION_TEST_SET: The reduction operation to use in the test set ("crop" or "resize").
    NUMBER_WORKERS: The number of processes to create for parallel computing.
    MAX_QUEUE_SIZE: The maximum size of the queue of preprocessed batches.
    PATH_LISTS: The path to the 3 lists of classes indexes and their corresponding names for each type of identification task.
    PATH_IMAGES: The path to give as prefix for every path stored in the dataset of paths towards images.
    PREFIX_RESULTS: The prefix of the path where to store the results.
    ADDITIONAL_INFORMATION: The additional information to write to the model description file.
    """
    
    SIZE_BATCH = 8
    NUMBER_EPOCHS = 2
    INITIAL_LEARNING_RATE = 0.001
    NUMBER_EPOCHS_LEARNING_RATE = 5
    DISCOUNT_FACTOR = 0.1
    WIDTH_IMAGE = 331
    HEIGHT_IMAGE = 331
    PROBABILITY_HORIZONTAL_FLIP = 0.5
    PROBABILITY_VERTICAL_FLIP = 0.0
    PROBABILITY_CROP_LEARNING_SET = 1.0
    REDUCTION_OPERATION_TEST_SET = "crop"
    NUMBER_WORKERS = multiprocessing.cpu_count()
    MAX_QUEUE_SIZE = 10
    PATH_LISTS = ""
    PATH_IMAGES = ""
    PREFIX_RESULTS = "Results/"
    ADDITIONAL_INFORMATION = "This model implements the ResNet50 neural network with weights initialized on imagenet. The last layer is a dense layer with a softmax activation function. All weights are directly trainable. The loss function is the categorical cross-entropy. The optimizer is the adam with the default beta1 and beta2 parameters."
    
    stringInformation = "SIZE_BATCH: {}\nNUMBER_EPOCHS: {}\nINITIAL_LEARNING_RATE: {}\nNUMBER_EPOCHS_LEARNING_RATE: {}\nDISCOUNT_FACTOR: {}\nWIDTH_IMAGE: {}\nHEIGHT_IMAGE: {}\nPROBABILITY_HORIZONTAL_FLIP: {}\nPROBABILITY_VERTICAL_FLIP: {}\nPROBABILITY_CROP_LEARNING_SET: {}\nREDUCTION_OPERATION_TEST_SET: {}\nNUMBER_WORKERS: {}\nMAX_QUEUE_SIZE: {}\nPATH_LISTS: {}\nPATH_IMAGES: {}\nPREFIX_RESULTS: {}\n\nADDITIONAL_INFORMATION:\n{}".format(SIZE_BATCH, NUMBER_EPOCHS, INITIAL_LEARNING_RATE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, WIDTH_IMAGE, HEIGHT_IMAGE, PROBABILITY_HORIZONTAL_FLIP, PROBABILITY_VERTICAL_FLIP, PROBABILITY_CROP_LEARNING_SET, REDUCTION_OPERATION_TEST_SET, NUMBER_WORKERS, MAX_QUEUE_SIZE, PATH_LISTS, PATH_IMAGES, PREFIX_RESULTS, ADDITIONAL_INFORMATION)
    
    data = pandas.read_csv("dataset.csv", sep=",", header=None)

    indexList = data.iloc[:,1].drop_duplicates().values

    X, y = create_pairs(data, indexList)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=356)
    
    model = create_neural_network(len(indexList), WIDTH_IMAGE, HEIGHT_IMAGE, INITIAL_LEARNING_RATE)
    
    learningSequence = fs.FragmentSequence(X_train, y_train, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES, PROBABILITY_CROP_LEARNING_SET, PROBABILITY_HORIZONTAL_FLIP, PROBABILITY_VERTICAL_FLIP)
    validationSequence = fsc.FragmentSequenceCentered(X_test, y_test, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES, REDUCTION_OPERATION_TEST_SET)
    
    currentTime = train_network(model, learningSequence, validationSequence, NUMBER_EPOCHS, NUMBER_WORKERS, SIZE_BATCH, INITIAL_LEARNING_RATE, MAX_QUEUE_SIZE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, PREFIX_RESULTS, stringInformation)
    
    
    parametersClass = ParametersClass(SIZE_BATCH, NUMBER_EPOCHS, INITIAL_LEARNING_RATE, NUMBER_EPOCHS_LEARNING_RATE, DISCOUNT_FACTOR, WIDTH_IMAGE, HEIGHT_IMAGE, PROBABILITY_HORIZONTAL_FLIP, PROBABILITY_VERTICAL_FLIP, PROBABILITY_CROP_LEARNING_SET, REDUCTION_OPERATION_TEST_SET, NUMBER_WORKERS, MAX_QUEUE_SIZE, PATH_LISTS, PATH_IMAGES, PREFIX_RESULTS, ADDITIONAL_INFORMATION)
    
    
    with open("{}/information_model_binary.pkl".format(PREFIX_RESULTS + currentTime), "wb") as f:
        pickle.dump(parametersClass, f)
    
    
