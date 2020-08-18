#################
# Python script to run the evaluation pipeline on a previously trained
# and saved ResNet50-Twin network.
#################

import keras
import os
from argparse import ArgumentParser

import EvaluationPipeline

if __name__ == "__main__":
    
    PATH_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
    PATH_CSV = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset.csv"
    PATH_MODEL = "/scratch/users/plnicolas/trainedmodels/xception_Random.h5"
    PREFIX_RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Results/Evaluation/ResNet50/"

    if not os.path.exists(PREFIX_RESULTS):
        os.makedirs(PREFIX_RESULTS)

    import tensorflow as tf
    model = keras.models.load_model(PATH_MODEL, custom_objects={"tf": tf})

    EvaluationPipeline.run_pipeline(model, PATH_CSV, PREFIX_RESULTS)