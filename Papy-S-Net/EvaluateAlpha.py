import keras

import EvaluationPipeline

if __name__ == "__main__":
    
    PATH_IMAGES = "/scratch/plnicolas/datasets/"
    PATH_CSV = "/home/plnicolas/codes/dataset_alpha.csv"
    PREFIX_RESULTS = "/home/plnicolas/codes/Results/Evaluation/Papy-S-Net/Alpha/"

    import tensorflow as tf
    model = keras.models.load_model('/scratch/plnicolas/trainedmodels/papyalpha.h5', custom_objects={"tf": tf})

    EvaluationPipeline.run_pipeline(model, PATH_CSV, PREFIX_RESULTS)