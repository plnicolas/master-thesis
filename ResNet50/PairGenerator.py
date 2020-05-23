import pandas
import random

def sample_pairs(K, data, IDList):
    """
    Function to create fragment pairs given a Pandas DataFrame and a list of IDs.

    Parameters:
    ----------
        - K: The number of pairs of each type (positive and negative) to sample. Duplicates will be dropped,
        so the final number of pairs WILL be smaller than 2K
        - Data: Pandas DataFrame with rows of the form [path_to_fragment_image, ID_of_original_papyrus]
        - IDList: List containing the IDs of the papyri to sample fragments from

    Returns:
    --------
        - pairs: A list of fragment pairs, of the form [path_to_frag1, path_to_frag2]
        - labels: A list of labels, i.e. original papyrus IDs

    """

    pairs = []
    labels = []

    # For each papyrus used for training
    for index in IDList:
        isIndex = data.iloc[:, 1] == index
        isNotIndex = data.iloc[:, 1] != index
        # List of images from the indexed papyrus
        indexTrueList = data[isIndex].iloc[:, 0]
        # List of images NOT from the indexed papyrus
        indexFalseList = data[isNotIndex].iloc[:, 0]

        # K negative pairs
        p1List = indexTrueList.sample(n=K, replace=True, random_state=356)
        p2List = indexFalseList.sample(n=K, replace=True, random_state=323)
        for k in range(K):
            pair = [p1List.values[k], p2List.values[k]]
            if pair not in pairs:
                pairs.append(pair)
                labels.append(1)

        # K positive pairs
        p1List = indexTrueList.sample(n=K, replace=True, random_state=362)
        p2List = indexTrueList.sample(n=K, replace=True, random_state=316)
        for k in range(K):
            pair = [p1List.values[k], p2List.values[k]]
            if pair not in pairs:
                pairs.append(pair)
                labels.append(0)

    # Shuffle the pairs and label lists before returning them
    # The two lists are shuffled at once with the same order, of course
    tmp = list(zip(pairs, labels))
    random.shuffle(tmp)
    pairs, labels = zip(*tmp)

    return pairs, labels


def create_pairs(K, pathCSV):
    """
    Function to create fragment pairs for training and testing.
    In order to avoid any bias, the papyri used to generate the training pairs are NOT used to generate the testing pairs.

    Parameters:
    ----------
        - K: The number of pairs of each type (positive and negative) to sample. Duplicates will be dropped,
        so the final number of pairs WILL be smaller than 2K
        - pathCSV: Path to a CSV file generated using retrieve_annotations_crop.py or retrieve_annotations_alphamask.py

    Returns:
    --------
        - The training pairs and their associated labels, the testing pairs and their associated labels.

    """

    # Load data from CSV file
    data = pandas.read_csv(pathCSV, sep=",", header=None)

    # IDList constains the ID of each papyrus
    IDList = data.iloc[:, 1].drop_duplicates().values
    # Number of papyri
    N = len(IDList)

    # IDRange is the proportion of papyri we will use to generate the training pairs
    # The remaining papyri will be used to generate the testing pairs =>
    # unbiased
    IDRange = (int)(N - (N / 4))
    print("IDRange: {}".format(IDRange))

    # To avoid any bias between training and testing, we partition the data
    # into a training dataset and a test dataset before sampling pairs
    trainData = data[data.iloc[:,1].isin(IDList[:IDRange])]
    testData = data[data.iloc[:,1].isin(IDList[IDRange:])]

    # Create the pairs
    pairsTrain, labelsTrain = sample_pairs(K, trainData, IDList[:IDRange])
    pairsTest, labelsTest = sample_pairs(K, testData, IDList[IDRange:])

    return pairsTrain, labelsTrain, pairsTest, labelsTest