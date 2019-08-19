# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import time

IMAGE_DIMS = (96, 96, 3)
args={"dataset":"..\\..\\..\\dataset_1" , 
      "model":'pokedex2.model',
       'labelbin':'lb2.pickle' }

data=[]
labels=[]

# load the image
imagePaths = sorted(list(paths.list_images(args["dataset"])))
# print(list(paths.list_images(args["dataset"])))


for imagePath in enumerate(imagePaths):
      start_time = time.time()
      image=cv2.imread(imagePath[1])
      image.resize(IMAGE_DIMS[0],IMAGE_DIMS[1],IMAGE_DIMS[2])
      image = img_to_array(image)
      data.append(image)
      # print(imagePaths[1])
      label = imagePath[1].split(os.path.sep)[-2]
      labels.append(label)
      end_time= time.time()
      print("Image {} is added. time: {}".format(imagePath[0],start_time-end_time))
      
      
      