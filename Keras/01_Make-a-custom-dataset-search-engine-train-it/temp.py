# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import load_model
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
import sys

IMAGE_DIMS = (96, 96, 3)
EPOCHS = 10
INIT_LR = 1e-3
BS = 32
args={"dataset":"..\\..\\..\\dataset_1" , 
      "model":'pokedex2.model',
       'labelbin':'lb2.pickle',
       'data_save_path':'..\\..\\..\\dataset_1\\data_label_np.pickle',
       'model_save_path':'..\\..\\..\\dataset_1\\pokedex.model',
       'binary_label':'..\\..\\..\\dataset_1\\lb.pickle',
       'plot':'..\\..\\..\\dataset_1\\Training Loss and Accuracy.png' }

data=[]
labels=[]

# load the image
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
# print(list(paths.list_images(args["dataset"])))

# Data_saved_path='..\\..\\..\\dataset_1\\data_label_np.pickle'
if not(os.path.exists(args["data_save_path"])):
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
            print("Image {} is added. (time: {:0.3f} sec)".format(imagePath[0],end_time-start_time))
      with open(args["data_save_path"], 'wb') as f:
            pickle.dump([data,labels], f)  
      print("[INFO] Data is saved as: \"data_label_np.pickle\" (Size:{:0.1f})".format(sys.getsizeof(args["data_save_path"])))
else:
      with open(args["data_save_path"], 'rb') as f:
            data, labels = pickle.load(f)
      print("[INFO] Data is Loaded: \"data_label_np.pickle\" (Size:{:0.1f})".format(sys.getsizeof(args["data_save_path"])))

# scale the raw pixel intensities to the range [0, 1]  
    
print(len(data),data[1].shape,"Max-Value in All images",np.max(data[:]))
data = np.array(data, dtype="float") / 255.0
print(len(data),data[1].shape,"Max-Value in All images",np.max(data[:]))
print("[INFO] The raw pixel intensities is scaled  to the range [0, 1]")

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print("[INFO] The classes: ",lb.classes_)

# Data partitioning 
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


if not(os.path.exists(args["model_save_path"])):
      # train the network
      print("[INFO] training network...")
      H = model.fit_generator(
            aug.flow(trainX, trainY, batch_size=BS),
            validation_data=(testX, testY),
            steps_per_epoch=len(trainX) // BS,
            epochs=EPOCHS, verbose=1)

      # save the model to disk
      print("[INFO] serializing network...")
      model.save(args["model_save_path"])
      # save the label binarizer to disk
      print("[INFO] serializing label binarizer...")
      f = open(args["binary_label"], "wb")
      f.write(pickle.dumps([H,lb]))
      f.close()
      print("[INFO] Model is saved as: \"pokedex.model\" (Size:{:0.1f})".format(sys.getsizeof(args["model_save_path"])))
      print("[INFO] Binary labels are saved as: \"lb.pickle\" (Size:{:0.1f})".format(sys.getsizeof(args["binary_label"])))
else:
      model = load_model(args["model_save_path"])
      H,lb = pickle.loads(open(args["binary_label"], "rb").read())
      print("[INFO] Model is loaded. (Size:{:0.1f})".format(sys.getsizeof(args["model_save_path"])))
      print("[INFO] Binary labels are loaded. (Size:{:0.1f})".format(sys.getsizeof(args["binary_label"])))
      
      
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])