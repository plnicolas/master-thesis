"""
    Python script to retrieve all papyrus annotations for the "blind test" fragments from Cytomine as regular crops and build a dataset
    (with CSV file) with them.
    Images will be saved in DOWNLOAD_PATH, while the CSV file will be saved at the root.
    
    Parameter: --rebuild
                "True" to rebuild the dataset and CSV file from scratch

    For security reasons, when rebuilding the dataset from scratch the already existing images are not deleted;
    this should be done manually beforehand (to avoid "dead" files -i.e. not used in the dataset- in the folders).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from argparse import ArgumentParser

import os
from os import path

import csv

from cytomine import Cytomine
from cytomine.models import AnnotationCollection
from cytomine.models.image import ImageInstanceCollection

PUBLIC_KEY = "d85b2768-5e54-45bf-8c00-76d2b512f1c5"
PRIVATE_KEY = "547db0e2-af21-4963-baff-df2afde92408"
HOST = "research.cytomine.be"
DOWNLOAD_PATH = "ImagesTest/"

if __name__ == '__main__':
    parser = ArgumentParser(prog="Annotation retrieval script for the blind test")
    parser.add_argument('--rebuild', dest='rebuild',
                        default='False', help="Set to true to rebuild the whole dataset")

    params, other = parser.parse_known_args(sys.argv[1:])

    print("Parameters: {}\n".format(params.rebuild))

    # Cytomine
    with Cytomine(host=HOST, public_key=PUBLIC_KEY, private_key=PRIVATE_KEY,
                  verbose=logging.INFO) as cytomine:
        annotations = AnnotationCollection()
        annotations.project = 163032268
        
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showGIS = True
        
        annotations.fetch()
        image_instances = ImageInstanceCollection().fetch_with_filter("project", 163032268)

        #CSV Writer object to build the CSV file of the dataset while retrieving the annotations
        #If we are rebuilding the dataset from scratch, rebuild the CSV file as well; else, append mode
        if(params.rebuild == "True"):
            c = csv.writer(open("dataset_test.csv", "w", newline="\n"))
        else:
            c = csv.writer(open("dataset_test.csv", "a", newline="\n"))

        for annotation in annotations:
            """
            print("ID: {} | Image: {} | Project: {} | Term: {} | User: {} | Area: {} | Perimeter: {} | WKT: {}".format(
                annotation.id,
                annotation.image,
                annotation.project,
                annotation.term,
                annotation.user,
                annotation.area,
                annotation.perimeter,
                annotation.location
            ))
            """

            annotationPath = DOWNLOAD_PATH + str(annotation.image) + "/" + str(annotation.id) + ".png"
            #If the annotation was already retrieved, skip
            if(not path.exists(annotationPath)):
                annotationFilename = ""
                for image in image_instances:
                    if image.id == annotation.image:
                        annotationFilename = image.filename

                annotationInfo = [annotationPath, annotationFilename]
                c.writerow(annotationInfo)
                annotation.dump(dest_pattern=os.path.join(DOWNLOAD_PATH, "{image}", "{id}.png"), mask=False, alpha=False)