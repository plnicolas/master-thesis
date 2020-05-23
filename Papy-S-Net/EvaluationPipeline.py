import numpy as np
import pandas

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.validation import check_symmetric
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

import keras
from keras import backend as K
import keras.utils
import keras.models
import keras.metrics
import datetime
import multiprocessing

import os
import random
import pickle

import PairGenerator

import FragmentSequence as fs
import FragmentSequenceValidation as fsv

################################
# SPECIFIC TO EACH ARCHITECTURE
################################

PAIRS = 4000
SIZE_BATCH = 16
NUMBER_EPOCHS = 40
INITIAL_LEARNING_RATE = 0.0001
NUMBER_EPOCHS_LEARNING_RATE = 20
DISCOUNT_FACTOR = 1
WIDTH_IMAGE = 128
HEIGHT_IMAGE = 128
PROBABILITY_HORIZONTAL_FLIP = 0.5
PROBABILITY_VERTICAL_FLIP = 0.5
MAX_QUEUE_SIZE = 50

#PATH_IMAGES = ""
PATH_IMAGES = "/scratch/plnicolas/datasets/"

###############################################
# EVERYTHING BELOW IS ARCHITECTURE INDEPENDENT
###############################################


def __test_IDs_list__(data):
    """
    Returns a list containing the IDs of the test papyri.
    This list can be used to sample a list of test fragment pairs.

    Parameters:
    ----------
        - data: Pandas DataFrame of a CSV file generated using retrieve_annotations_crop.py or retrieve_annotations_alphamask.py

    Returns:
    --------
        - A list containing the IDs of the test papyri

    """

    # IDList constains the ID of each papyrus
    IDList = data.iloc[:, 1].drop_duplicates().values
    # Number of papyri
    N = len(IDList)

    # IDRange is the proportion of papyri used to generate the training pairs
    # The remaining papyri wered used to generate the testing pairs
    IDRange = (int)(N - (N / 4))
    
    testIDs = IDList[IDRange:]

    return testIDs


def __test_pairs__(data):
    """
    Function to create all test fragment pairs given the test data's Pandas DataFrame.
    If there are k test fragments, creates k^2 pairs.

    Parameters:
    ----------
        - data: Pandas DataFrame with rows of the form [path_to_fragment_image, ID_of_original_papyrus]

    Returns:
    --------
        - pairs: The list of all test fragment pairs, of the form [path_to_frag1, path_to_frag2]
        - labels: The list of labels, i.e. original papyrus IDs

    """

    pairs = []
    labels = []

    for row in data.values:
        frag = row[0]
        ID = row[1]

        for row2 in data.values:
            frag2 = row2[0]
            ID2 = row2[1]

            pairs.append([frag, frag2])
            if ID == ID2:
                labels.append(0)
            else:
                labels.append(1)

    return pairs, labels

def get_distance_matrix(y_pred, N, pairs, labels):
    """
    Get a distance matrix based on a model's predictions for a list of test fragment pairs.

    Parameters:
    ----------
        - y_pred: Output given by the Keras model to evaluate on the test pairs.
        - N: The number of test fragments.
        - pairs: A list containing all test fragment pairs.
        - labels: The corresponding list of labels.

    Returns:
    --------
        - distanceMatrix: The distance matrix based on the model's predictions
    """

    # Only keep the predicted probability for class 0 (=similar)
    distanceVector = []
    for i in y_pred:
        # The distance is one minus the probability assigned to class 0 (= similar)
        # The higher the probability of the fragments being similar, the lower the distance between them
        distance = 1 - i[0]
        distanceVector.append(distance)

    # Convert the distance vector to a distance matrix
    distanceMatrix = np.ndarray(shape=(N,N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            distanceMatrix[i][j] = distanceVector[(i * N) + j]

    print(distanceMatrix)
    return distanceMatrix

def run_TSNE(data, distanceMatrix, pathResults):

    embeddings = TSNE(n_components=2, random_state=323, metric="precomputed", n_jobs=-1).fit_transform(distanceMatrix)

    # Convert the embeddings array to a DataFrame to plot easily with color code 
    d = {'x': embeddings[:,0], 'y': embeddings[:,1], 'Papyrus': data.values[:,1]}

    embeddingsDF = pandas.DataFrame(data=d)

    #Plot TSNE
    graph = sns.scatterplot(x='x', y='y', hue='Papyrus', data=embeddingsDF)
    plt.title("TSNE")
    plt.tight_layout()
    fig = graph.get_figure()
    fig.savefig('{}TSNE.png'.format(pathResults)) 
    plt.clf()

def run_MDS(data, distanceMatrix, pathResults):

    embeddings = MDS(n_components=2, random_state=323, dissimilarity="precomputed", n_jobs=-1).fit_transform(distanceMatrix)

    # Convert the embeddings array to a DataFrame to plot easily with color code 
    d = {'x': embeddings[:,0], 'y': embeddings[:,1], 'Papyrus': data.values[:,1]}

    embeddingsDF = pandas.DataFrame(data=d)

    #Plot MDS
    graph = sns.scatterplot(x='x', y='y', hue='Papyrus', data=embeddingsDF)
    plt.title("MDS")
    plt.tight_layout()
    fig = graph.get_figure()
    fig.savefig('{}MDS.png'.format(pathResults))
    plt.clf()

def print_classification_report(y_pred, y_true):
    """
    Print the classification report given some predictions.

    Parameters:
    ----------
        - y_pred: The predictions of some model.
        - y_true: The true labels.

    Returns:
    --------
        - /

    """

    # Argmax because we want the class (= index), not the probability of the input belonging to the class
    #(for ROC curve or Precision-Recall curve, take the value at index ???? instead of the argmax)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(y_true, y_pred_bool))


def plot_curves(y_pred, y_true, N, pathResults):

    # Only keep the predicted probability for class 0 (=similar)
    # /!\ This means y_pred has values in [0;1] and the positive class is no longer 0,
    # but 1 !
    y_pred = y_pred[:, 0]

    # Convert the prediction vector to a prediction matrix
    # 1 line = predictions for 1 fragment vs others
    # Same for true labels
    predictionMatrix = np.ndarray(shape=(N,N), dtype=np.float64)
    labelMatrix = np.ndarray(shape=(N,N))
    for i in range(N):
        for j in range(N):
            predictionMatrix[i][j] = y_pred[(i * N) + j]
            labelMatrix[i][j] = y_true[(i * N) + j]

    precisionsList = []
    recallsList =  []
    auprList = []
    fprList = []
    tprList = []
    aucList = []

    # For each fragment
    for fragment in range(N):
        # Prediction and true label vs all other fragments
        predLabels = predictionMatrix[fragment]
        trueLabels = labelMatrix[fragment]

        # Compute precisions and recalls for different thresholds
        # Positive label is 1 because y_pred has probabilities 
        precisions, recalls, thresholds = precision_recall_curve(trueLabels, predLabels, pos_label=1)
        aupr = average_precision_score(trueLabels, predLabels, pos_label=1)
        fpr, tpr, thresholds = roc_curve(trueLabels, predLabels, pos_label=1)
        auc = roc_auc_score(trueLabels, predLabels)
        

        # Store into a list
        precisionsList.append(precisions)
        recallsList.append(recalls)
        auprList.append(aupr)
        fprList.append(fpr)
        tprList.append(tpr)
        aucList.append(auc)

    # Create DataFrame to easily compute mean vectors
    precisionDF = pandas.DataFrame(precisionsList)
    recallsDF = pandas.DataFrame(recallsList)
    fprDF = pandas.DataFrame(fprList)
    tprDF = pandas.DataFrame(tprList)

    # Compute means
    meanPrecisions = precisionDF.mean().values
    meanRecalls = recallsDF.mean().values
    meanfpr = fprDF.mean().values
    meantpr = tprDF.mean().values


    plt.plot(meanRecalls, meanPrecisions, color='darkorange', label='PR curve (mean MAP = %0.2f)' % np.mean(auprList))
    plt.title("Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("{}meanPR.png".format(pathResults))
    plt.clf()

    plt.plot(meanfpr, meantpr, color='darkorange', label='ROC curve (mean AUC = %0.2f)' % np.mean(aucList))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("{}meanROC.png".format(pathResults))
    plt.clf()



def run_pipeline(model, pathCSV, pathResults):
    """
    Run the complete evaluation pipeline.


    Parameters:
    ----------
        - model: The model to be evaluated, already trained.
        - pathCSV: Path to the CSV file of the dataset.
        - pathResults: Path to the folder where the results (figures) have to be saved

    Returns:
    --------
        - /

    """

    # Load data from CSV file
    data = pandas.read_csv(pathCSV, sep=",", header=None)

    # Get list of test IDs
    testIDList = __test_IDs_list__(data)

    # Keep only test fragments
    testData = data[data.iloc[:,1].isin(testIDList)]

    # Replace papyrus IDs with "Papyrus 1", "Papyrus 2" etc
    stringList = ["Papyrus 1", "Papyrus 2", "Papyrus 3", "Papyrus 4", "Papyrus 5", "Papyrus 6"]
    testData = testData.replace(testIDList, stringList)

    #Generate all possible pairs of test fragments
    pairs, labels = __test_pairs__(testData)

    #Get distance matrix of the test fragments
    numberOfFragments = testData.values.shape[0]
    
    # Create keras Sequence using the test data
    testSequence = fsv.FragmentSequenceValidation(pairs, labels, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)

    # Get prediction of the model for each fragment pair
    y_pred = model.predict_generator(testSequence, max_queue_size=MAX_QUEUE_SIZE, workers=multiprocessing.cpu_count(), use_multiprocessing=True)

    # Print the global classification report
    print_classification_report(y_pred, labels)

    # Get the distance matrix from the predictions
    distanceMatrix = get_distance_matrix(y_pred, numberOfFragments, pairs, labels)
    # Enforces symmetry (necessary because of floating-point rounding errors)
    distanceMatrix = check_symmetric(distanceMatrix)

    # Run TSNE and MDS and plot results
    run_TSNE(testData, distanceMatrix, pathResults)
    run_MDS(testData, distanceMatrix, pathResults)
    plot_curves(y_pred, labels, numberOfFragments, pathResults)