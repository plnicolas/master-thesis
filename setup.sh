#!/bin/bash

conda create --name deeplearning --file requirements.txt

conda activate deeplearning

curl -s https://packagecloud.io/install/repositories/cytomine-uliege/Cytomine-python-client/script.python.sh | bash
pip install cytomine-python-client