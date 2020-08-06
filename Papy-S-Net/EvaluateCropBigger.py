import keras

import EvaluationPipelineBigger

if __name__ == "__main__":
    
    PATH_IMAGES = "/scratch/plnicolas/datasets/"
    PATH_CSV = "/home/plnicolas/codes/dataset.csv"
    PREFIX_RESULTS = "/home/plnicolas/codes/Results/Evaluation/Papy-S-Net/CropBigger/"

    import tensorflow as tf
    model = keras.models.load_model('/scratch/plnicolas/trainedmodels/papycropbigger.h5', custom_objects={"tf": tf})

    EvaluationPipelineBigger.run_pipeline(model, PATH_CSV, PREFIX_RESULTS)