from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from argparse import ArgumentParser

import os

import csv

from cytomine import Cytomine
from cytomine.models import AnnotationCollection

PUBLIC_KEY = "d85b2768-5e54-45bf-8c00-76d2b512f1c5"
PRIVATE_KEY = "547db0e2-af21-4963-baff-df2afde92408"
HOST = "research.cytomine.be"
DOWNLOAD_PATH = "Images/"

if __name__ == '__main__':

    # Cytomine
    with Cytomine(host=HOST, public_key=PUBLIC_KEY, private_key=PRIVATE_KEY,
                  verbose=logging.INFO) as cytomine:
        annotations = AnnotationCollection()
        annotations.project = 136266234
        
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showGIS = True
        
        annotations.fetch()
        #print(annotations)

        #CSV Writer object to build the CSV file of the dataset while retrieving the annotations
        c = csv.writer(open("dataset.csv", "w", newline="\n"))

        i = 0

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
            annotationPath = DOWNLOAD_PATH + str(annotation.image) + "/crop/" + str(annotation.id) + ".png"
            annotationInfo = [annotationPath, annotation.image]
            c.writerow(annotationInfo)

            annotation.dump(dest_pattern=os.path.join(DOWNLOAD_PATH, "{image}", "crop", "{id}.png"), mask=False, alpha=False)