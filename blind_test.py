import numpy as np 
import pandas
import keras
from argparse import ArgumentParser
import multiprocessing

import FragmentSequenceValidation as fsv
import EvaluationPipeline

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
    PATH_CSV = "dataset_test.csv"

    # Load data from CSV file
    data = pandas.read_csv(PATH_CSV, sep=",", header=None)

    # Generate all possible pairs of test fragments
    pairs, labels, pairsFilename = __test_pairs__(data)

    import tensorflow as tf
    model = keras.models.load_model("model_trained.h5", custom_objects={"tf": tf})

    # Create keras Sequence using the test data
    testSequence = fsv.FragmentSequenceValidation(pairs, labels, SIZE_BATCH, WIDTH_IMAGE, HEIGHT_IMAGE, PATH_IMAGES)
    # Get prediction of the model for each fragment pair
    y_pred = model.predict_generator(testSequence, max_queue_size=MAX_QUEUE_SIZE, workers=multiprocessing.cpu_count(), use_multiprocessing=True, verbose=1)

    predictions = y_pred[:,0]
    predictions = np.where(predictions > 0.5, 1, 0)

    DFpair = pandas.DataFrame(np.array(pairsFilename))
    DFpred = pandas.DataFrame(predictions)
    print(DFpair)

    DF = pandas.concat([DFpair, DFpred], axis=1)
    print(DF)
    DF.to_csv("predictions.csv", index=False)


    