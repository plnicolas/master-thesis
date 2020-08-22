#################
# Python script to run the evaluation pipeline on a previously trained
# and saved Papy-S-Net network.
#################

import keras
import os
from argparse import ArgumentParser

import EvaluationPipeline

def get_arguments():
    # Get the arguments of the program
    parser = ArgumentParser(prog="Papy-S-Net architecture evaluation")

    parser.add_argument('--size', dest='size', default=128, type=int, help="Image size (square; only one value needed)")
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--brightness', dest='brightness', default=0, type=int, help="Brightness shifts during training (0 = no)")

    return parser.parse_args()

if __name__ == "__main__":

    args = get_arguments()

    PATH_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
    PATH_CSV = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset.csv"
    PATH_MODEL = "/scratch/users/plnicolas/trainedmodels/papycrop.h5"
    PREFIX_RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Results/Evaluation/Papy-S-Net/Crop/"

    if not os.path.exists(PREFIX_RESULTS):
        os.makedirs(PREFIX_RESULTS)

    import tensorflow as tf
    model = keras.models.load_model(PATH_MODEL, custom_objects={"tf": tf})

    EvaluationPipeline.run_pipeline(model, PATH_CSV, args, PREFIX_RESULTS)