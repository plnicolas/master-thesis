import numpy as np 
import pandas
import keras
from argparse import ArgumentParser
import multiprocessing

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.validation import check_symmetric
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

import FragmentSequenceValidation as fsv
import EvaluationPipeline

def __test_pairs__(data):
    """
    Function to create all test fragment pairs given the test data's Pandas DataFrame,
    except trivial ones and "duplicates" (like {Fragment 1, Fragment 1} or having both
    {Fragment 1, Fragment 2} and {Fragment 2, Fragment 1}).
    Thus the list of pairs is smaller and more convenient for manually checking the results.

    Parameters:
    ----------
        - data: Pandas DataFrame with rows of the form [path_to_fragment_image, ID_of_original_papyrus]

    Returns:
    --------
        - pairs: The list of all test fragment pairs, of the form [path_to_frag1, path_to_frag2]
        - labels: The list of labels, i.e. original papyrus IDs
        - pairsFilename: The list of all test fragment pairs, but with the filenames instead of the IDs

    """

    pairs = []
    labels = []
    pairsFilename = []

    for row in data.values:
        frag = row[0]
        ID = row[1]

        for row2 in data.values:
            frag2 = row2[0]
            ID2 = row2[1]
            if [frag, frag2] not in pairs:
                if [frag2, frag] not in pairs:
                    if frag != frag2:
                        pairs.append([frag, frag2])
                        pairsFilename.append([ID, ID2])
                        if ID == ID2:
                            labels.append(0)
                        else:
                            labels.append(1)

    return pairs, labels, pairsFilename


def __test_pairs_full__(data):
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

    return distanceMatrix

def get_arguments():
    # Get the arguments of the program
    parser = ArgumentParser(prog="Papy-S-Net architecture applied to the blind test")

    parser.add_argument('--size', dest='size', default=128, type=int, help="Image size (square; only one value needed)")
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--brightness', dest='brightness', default=0, type=int, help="Brightness shifts during training (0 = no)")

    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_arguments()

    SIZE_BATCH = args.batch_size
    WIDTH_IMAGE = args.size
    HEIGHT_IMAGE = args.size
    MAX_QUEUE_SIZE = 50

    PATH_IMAGES = ""
    #PATH_IMAGES = "/scratch/users/plnicolas/datasets/"
    PATH_CSV = "dataset_test.csv"

    # Load data from CSV file
    data = pandas.read_csv(PATH_CSV, sep=",", header=None)

    import tensorflow as tf
    model = keras.models.load_model("resnet_IN.h5", custom_objects={"tf": tf})

    # # # # # # # # #
    # List for Mr. Polis
    # # # # # # # # #

    # Generate all possible pairs of test fragments
    pairs, labels, pairsFilename = __test_pairs__(data)

    # Create keras Sequence using the test data
    testSequence = fsv.FragmentSequenceValidation(pairs, labels, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)
    # Get prediction of the model for each fragment pair
    y_pred = model.predict_generator(testSequence, max_queue_size=MAX_QUEUE_SIZE, workers=multiprocessing.cpu_count(), use_multiprocessing=True, verbose=1)

    predictions = y_pred[:,0]
    predictions = np.digitize(predictions, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    DFpair = pandas.DataFrame(np.array(pairsFilename), columns=["Fragment 1", "Fragment 2"])
    DFpred = pandas.DataFrame(predictions, columns=["Similarity"])

    DF = pandas.concat([DFpair, DFpred], axis=1)
    DF = DF.sort_values(by=["Similarity"], ascending=False)
    DF.to_csv("predictions.csv", index=False)

    # Histogram of the predictions
    hist = np.histogram(DF["Similarity"].values, bins=10)
    fig, ax = plt.subplots()
    offset = .4
    plt.bar(hist[1][1:],hist[0])
    ax.set_xticks(hist[1][1:] + offset)
    ax.set_xticklabels( ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9') )
    plt.xlabel("Score")
    plt.savefig('histogram.png')
    plt.clf()

    # # # # # # # # #
    # To generate the similarity matrix
    # # # # # # # # #

    # Generate all possible pairs of test fragments
    pairs, labels = __test_pairs_full__(data)

    # Create keras Sequence using the test data
    testSequence = fsv.FragmentSequenceValidation(pairs, labels, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)
    # Get prediction of the model for each fragment pair
    y_pred = model.predict_generator(testSequence, max_queue_size=MAX_QUEUE_SIZE, workers=multiprocessing.cpu_count(), use_multiprocessing=True, verbose=1)

    numberOfFragments = data.values.shape[0]

    # Get the distance matrix from the predictions
    distanceMatrix = get_distance_matrix(y_pred, numberOfFragments, pairs, labels)
    distanceMatrix = check_symmetric(distanceMatrix)

    distanceMatrix = 1 - distanceMatrix

    plt.figure(figsize=(2000,2000))
    plt.matshow(distanceMatrix)
    plt.xticks(range(data.values.shape[0]), data.values[:,1], fontsize=1.5, rotation=90)
    plt.yticks(range(data.values.shape[0]), data.values[:,1], fontsize=1.5)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('matrix.png', dpi=500)