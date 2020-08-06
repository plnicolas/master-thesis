import numpy as np
import pandas

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.validation import check_symmetric
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

from scipy import interp

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

import sys
from skimage import io
from skimage import exposure
import cv2

import keras

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
INITIAL_LEARNING_RATE = 0.01
NUMBER_EPOCHS_LEARNING_RATE = 20
DISCOUNT_FACTOR = 0.1
WIDTH_IMAGE = 224
HEIGHT_IMAGE = 224
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
        distance = i
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
    #y_pred = y_pred[:, 0]

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
    base_fpr = np.linspace(0, 1, 101)
    base_rec = np.linspace(0, 1, 101)

    # For each fragment
    for fragment in range(N):
        # Prediction and true label vs all other fragments
        predLabels = predictionMatrix[fragment]
        trueLabels = labelMatrix[fragment]

        # Compute precisions and recalls for different thresholds
        # Positive label is 1 because y_pred has probabilities 
        precisions, recalls, thresholds = precision_recall_curve(trueLabels, predLabels, pos_label=0)
        aupr = average_precision_score(trueLabels, predLabels, pos_label=0)
        fpr, tpr, thresholds = roc_curve(trueLabels, predLabels, pos_label=0)
        AUC = auc(fpr, tpr)
        
        # Interpolate TPRs (correct way to build mean ROC curve)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0

        # Interpolate precisions
        # First sort recalls in ascending order (necessary for interpolation) while retaining order of pairings with precisions
        recalls, precisions = zip(*sorted(zip(recalls, precisions)))
        precisions = interp(base_rec, recalls, precisions)
        precisions[0] = 1.0

        # Store into a list
        precisionsList.append(precisions)
        auprList.append(aupr)
        tprList.append(tpr)
        aucList.append(AUC)

    # ROC
    tprs = np.array(tprList)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucList)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b', alpha = 0.8, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.tight_layout()
    plt.savefig("{}meanROC.png".format(pathResults))
    plt.clf()

    # Precision-Recall

    precs = np.array(precisionsList)
    mean_precs = precs.mean(axis=0)
    std = precs.std(axis=0)

    mean_map = auc(base_rec, mean_precs)
    std_map = np.std(auprList)

    precs_upper = np.minimum(mean_precs + std, 1)
    precs_lower = mean_precs - std

    plt.plot(base_rec, mean_precs, 'b', alpha = 0.8, label=r'Mean PR (MAP = %0.2f $\pm$ %0.2f)' % (mean_map, std_map),)
    plt.fill_between(base_rec, precs_lower, precs_upper, color = 'blue', alpha = 0.2)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc="best")
    plt.title('Precision-Recall curve')
    plt.tight_layout()
    plt.savefig("{}meanPR.png".format(pathResults))
    plt.clf()


def plot_dendrogram(model, data, pathResults, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    d = dendrogram(linkage_matrix, **kwargs)
    
    # Create a color palette with 6 color for the 6 test papyri
    my_palette = plt.cm.get_cmap("Dark2", 6)

    # transforme the Papyrus column in a categorical variable. It will allow to put one color on each level.
    labels = np.array(d.get("ivl"))

    from sklearn.preprocessing import LabelEncoder  
    le = LabelEncoder()
    my_color = le.fit_transform(labels)

    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    num=-1
    for lbl in xlbls:
        num += 1
        val = my_color[num]
        lbl.set_color(my_palette(val))
    plt.xlabel("Fragment")
    plt.tight_layout()
    plt.savefig("{}dendrogram.png".format(pathResults))
    plt.clf()

"""
# Function to calculate Chi-distace 
def chi2_distance(A, B): 
  
    # compute the chi-squared distance using above formula 
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)  
                      for (a, b) in zip(A, B)]) 
  
    return chi 

def colour_histogram_dist(imagePairs):

    y_pred = []
    j = 0
    # For each pair
    for i in imagePairs:
        image1Path = i[0]
        image2Path = i[1]
        # Load image and convert it to RGB (OpenCV stores images in BGR format)
        image1 = io.imread(image1Path)
        image2 = io.imread(image2Path)
        if j < 1:
            histr1, bins = np.histogram(image1[:,:,0], bins=np.arange(256))
            histg1, bins = np.histogram(image1[:,:,1], bins=np.arange(256))
            histb1, bins = np.histogram(image1[:,:,2], bins=np.arange(256))
            plt.subplot(2,1,1)
            plt.plot(histr1, color="red")
            plt.plot(histg1, color="green")
            plt.plot(histb1, color="blue")

            histr2, bins = exposure.histogram(image2[:,:,0], nbins=2)
            histg2, bins = exposure.histogram(image2[:,:,1], nbins=2)
            histb2, bins = exposure.histogram(image2[:,:,2], nbins=2)
            plt.subplot(2,1,2)
            #plt.plot(histr2, color="red")
            #plt.plot(histg2, color="green")
            #plt.plot(histb2, color="blue")

            #plt.show()

            print(histr1)
            print(histr2)
            print(chi2_distance(histr1, histr2))

            img = cv2.imread(image1Path, -1)
            color = ('b','g','r')
            for channel,col in enumerate(color):
                histr = cv2.calcHist([img],[channel],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.title('Histogram for color scale picture')
            plt.show()


        H1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        H1 = cv2.normalize(H1, H1).flatten()

        H2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        H2 = cv2.normalize(H2, H2).flatten()

        # Compare the histogrames using the Chi-Squared distance
        y_pred.append(cv2.compareHist(H1, H2, cv2.HISTCMP_CHISQR))

    print(y_pred[:20])

    return y_pred
"""

def colour_histogram_dist(imagePairs):

    y_pred = []
    j = 0
    # For each pair
    for i in imagePairs:
        image1Path = i[0]
        image2Path = i[1]
        # Load image and convert it to RGB (OpenCV stores images in BGR format)
        image1 = cv2.imread(image1Path)
        image2 = cv2.imread(image2Path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        H1 = cv2.calcHist([image1], [0, 1, 2], None, [2, 2, 2], [0, 256, 0, 256, 0, 256])
        H1 = cv2.normalize(H1, H1).flatten()

        H2 = cv2.calcHist([image2], [0, 1, 2], None, [2, 2, 2], [0, 256, 0, 256, 0, 256])
        H2 = cv2.normalize(H2, H2).flatten()

        # Compare the histogrames using the Chi-Squared distance
        y_pred.append(cv2.compareHist(H1, H2, cv2.HISTCMP_CHISQR))

    print(y_pred[:20])

    return y_pred

def run_pipeline(pathCSV, pathResults):
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

    numberOfFragments = testData.values.shape[0]

    # Create keras Sequence using the test data
    # testSequence = fsv.FragmentSequenceValidation(pairs, labels, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)

    # Get prediction of the colour-histogram baseline for each fragment pair
    y_pred = colour_histogram_dist(pairs)

    # Get the distance matrix from the predictions
    distanceMatrix = get_distance_matrix(y_pred, numberOfFragments, pairs, labels)
    distanceMatrix = check_symmetric(distanceMatrix)

    # Run TSNE and MDS and plot results
    run_TSNE(testData, distanceMatrix, pathResults)
    run_MDS(testData, distanceMatrix, pathResults)

    # y_pred must have values in [0,1] to pass to plot_curves
    y_pred = np.array(y_pred)
    y_pred = y_pred / np.max(y_pred)
    y_pred = 1 - y_pred
    plot_curves(y_pred, labels, numberOfFragments, pathResults)

    # setting distance_threshold=0 ensures we compute the full tree.
    modelClustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    modelClustering = modelClustering.fit(distanceMatrix)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(modelClustering, testData, pathResults, truncate_mode=None, labels=testData.iloc[:,1].values)


if __name__ == '__main__':

    PATH_CSV = "dataset.csv"
    PREFIX_RESULTS = "Results/"

    run_pipeline(PATH_CSV, PREFIX_RESULTS)