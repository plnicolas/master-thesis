import keras

import os
from argparse import ArgumentParser

import EvaluationPipeline

if __name__ == "__main__":
    
    PATH_IMAGES = "/scratch/users/plnicolas/datasets/"
    PATH_CSV = "/home/plnicolas/codes/dataset.csv"
    PREFIX_RESULTS = "/home/plnicolas/codes/Results/Evaluation/Xception/Random/"

    if not os.path.exists(PREFIX_RESULTS):
        os.makedirs(PREFIX_RESULTS)

    import tensorflow as tf
    model = keras.models.load_model('/scratch/users/plnicolas/trainedmodels/xception_Random.h5', custom_objects={"tf": tf})

    EvaluationPipeline.run_pipeline(model, PATH_CSV, PREFIX_RESULTS)