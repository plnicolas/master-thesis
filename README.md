# Papyrus matching using machine learning

## Main requirements
- Python 3.6
- Keras 2.2.4 with Tensorflow backend (should be Tensorflow 2.0-compatible with minor changes)
- The [**Cytomine** python client](https://github.com/cytomine/Cytomine-python-client) is required to download the dataset from the servers of Cytomine.

## Installation
- A suitable Anaconda environment can be created using the requirements.txt file at the root of the project, for convenience.
- To download the dataset from the Cytomine server, at the root of the project, use the command:
  - `python retrieve_annotations_crop.py`
  - This will download the dataset at the root of the project and generate a `dataset.csv` file.
- The dataset can also be downloaded as alphamasks using the command:
  - `python retrieve_annotations_alphamask.py`
  - The data will be in the same folder as the base dataset, and a `dataset_alpha.csv` file will be generated.
 
:exclamation: Most of the alphamasks will be identical to the crops, and the first experiment (crops vs alphamasks) can't be replicated as-is, as it was done on a previous iteration of the dataset (cf. PDF report) were all fragments had complex shapes, which is NOT the case anymore. Therefore, the alphamask-related code is mostly left here for exhaustivity.

## Experiment with the ResNet & Xception architectures
### _ResNet50-Twin:_
The code is in the `./ResNet50` folder; the main programs are `model_resnet50_IN.py` and `model_resnet50_Random.py`.\
From the root of the project:
- `python ResNet50/model_resnet50_IN.py` will run the network with pre-trained weights
- `python ResNet50/model_resnet50_Random.py` will run the network with random weight initialization

### _Xception-Twin:_
The code is in the `./Xception/` folder; the main programs are `model_xception_IN.py` and `model_xception_Random.py`.\
From the root of the project:
- `python Xception/model_xception_IN.py` will run the network with pre-trained weights
- `python Xception/model_xception_Random.py` will run the network with random weight initialization

## Experiment with the Papy-S-Net architecture
The code is in the `./Papy-S-Net/` folder; the main program is `model_papysnet.py` (`model_papysnet_alpha.py` for the alphamask version).\
Several optional parameters can be specified when launching the program:
- --size to specify the input image size (128 by default)
- --batch_size to specify the batch size (16 by default)
- --brightness to specify whether or not to use random shits in brightness (0 for no shifts)

Example of usage (from the root of the project):
- `python Papy-S-Net/model_papysnet --size 224 --batch-size 32 --brightness 0`
