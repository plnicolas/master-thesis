# Papyrus matching using machine learning

## Main requirements
- Python 3.6
- Keras 2.2.4 with Tensorflow backend (should be Tensorflow 2.0-compatible with minor changes)
- The [**Cytomine** python client](https://github.com/cytomine/Cytomine-python-client) is required to download the dataset from the servers of Cytomine.

## Installation
- A suitable Anaconda environment can be created using the requirements.txt file at the root of the project, for convenience.
- To download the dataset from the Cytomine server, at the root of the project, use the command:
  - `python retrieve_annotations_crop.py`
  - This will download the dataset at the root of the project.
