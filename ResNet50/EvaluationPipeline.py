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
WIDTH_IMAGE = 224
HEIGHT_IMAGE = 224
MAX_QUEUE_SIZE = 50

PATH_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
#PATH_IMAGES = "/scratch/users/plnicolas/datasets/"

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
    
    # To have readable figures for the report, only 10 test papyri
    # The last 10 papyri, which have not been seen during training
    testIDs = IDList[-10:]
    #testIDs = IDList[IDRange:]

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
    """
    Run the t-SNE algorithm using the distance matrix obtained from the predictions of a model.

    Parameters:
    ----------
        - data: The test data.
        - distanceMatrix: A symmetric distance matrix obtained from the predictions of a model.
        - pathResults: Path to the folder where the results have to be saved

    Returns:
    --------
        - /
    """

    embeddings = TSNE(n_components=2, random_state=323, metric="precomputed", n_jobs=-1).fit_transform(distanceMatrix)

    # Convert the embeddings array to a DataFrame to plot easily with color code 
    d = {'x': embeddings[:,0], 'y': embeddings[:,1], 'Papyrus': data.values[:,1]}

    embeddingsDF = pandas.DataFrame(data=d)

    #Plot TSNE
    graph = sns.scatterplot(x='x', y='y', hue='Papyrus', data=embeddingsDF, legend=False)
    plt.title("TSNE")
    plt.tight_layout()
    fig = graph.get_figure()
    fig.savefig('{}TSNE.png'.format(pathResults)) 
    plt.clf()

def run_MDS(data, distanceMatrix, pathResults):
    """
    Run the MDS algorithm using the distance matrix obtained from the predictions of a model.

    Parameters:
    ----------
        - data: The test data.
        - distanceMatrix: A symmetric distance matrix obtained from the predictions of a model.
        - pathResults: Path to the folder where the results have to be saved

    Returns:
    --------
        - /
    """

    embeddings = MDS(n_components=2, random_state=323, dissimilarity="precomputed", n_jobs=-1).fit_transform(distanceMatrix)

    # Convert the embeddings array to a DataFrame to plot easily with color code 
    d = {'x': embeddings[:,0], 'y': embeddings[:,1], 'Papyrus': data.values[:,1]}

    embeddingsDF = pandas.DataFrame(data=d)

    #Plot MDS
    graph = sns.scatterplot(x='x', y='y', hue='Papyrus', data=embeddingsDF, legend=False)
    plt.title("MDS")
    plt.tight_layout()
    fig = graph.get_figure()
    fig.savefig('{}MDS.png'.format(pathResults))
    plt.clf()

def plot_curves(y_pred, y_true, N, pathResults):
    """
    Create ROC and MDS curves on the test pairs from the predictions of a model.

    Parameters:
    ----------
        - y_pred: The model's predicted labels.
        - y_true: The true labels.
        - N: The number of distinct fragments in the test data
        - pathResults: Path to the folder where the results have to be saved

    Returns:
    --------
        - /
    """

    # Only keep the predicted probability for class 0 (=similar)
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
    """
    Create linkage matrix and then plot the dendrogram.

    Parameters:
    ----------
        - model: A fitted AgglomerativeClustering model.
        - data: The test data
        - pathResults: Path to the folder where the result (dendrograms figure) has to be saved
        - kwargs: Additional parameters specific to the dendrogram function of scipy

    Returns:
    --------
        - /
    """

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
    
    # Create a color palette with 10 color for the 10 test papyri
    my_palette = plt.cm.get_cmap("tab10", 10)

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
    stringList = []
    i = 1
    while i <= len(testIDList):
        tmp = "Papyrus {}".format(i)
        stringList.append(tmp)
        i += 1

    testData = testData.replace(testIDList, stringList)

    #Generate all possible pairs of test fragments
    pairs, labels = __test_pairs__(testData)

    numberOfFragments = testData.values.shape[0]

    # Create keras Sequence using the test data
    testSequence = fsv.FragmentSequenceValidation(pairs, labels, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)

    # Get prediction of the model for each fragment pair
    y_pred = model.predict_generator(testSequence, max_queue_size=MAX_QUEUE_SIZE, workers=multiprocessing.cpu_count(), use_multiprocessing=True, verbose=1)

    # Get the distance matrix from the predictions
    distanceMatrix = get_distance_matrix(y_pred, numberOfFragments, pairs, labels)
    distanceMatrix = check_symmetric(distanceMatrix)

    # Run TSNE and MDS and plot results
    run_TSNE(testData, distanceMatrix, pathResults)
    run_MDS(testData, distanceMatrix, pathResults)
    plot_curves(y_pred, labels, numberOfFragments, pathResults)

    # setting distance_threshold=0 ensures we compute the full tree.
    modelClustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    modelClustering = modelClustering.fit(distanceMatrix)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(modelClustering, testData, pathResults, truncate_mode=None, labels=testData.iloc[:,1].values)