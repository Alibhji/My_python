import matplotlib.pyplot as plt
import numpy as np
import os, sys
import tensorflow as tf
print(sys.version)

ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
                    2.71054693,  5.12469308,  # anchor box 2, width,  height
                   10.47181473, 10.09646365,  # anchor box 3, width,  height
                    5.48531347,  8.11011331]) # anchor box 4, width,  height

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']

### The location where the VOC2012 data is saved.
train_image_folder_2012 = "/home/ali_bhji/dataset/VOCdevkit/VOC2012/JPEGImages/"
train_annot_folder_2012 = "/home/ali_bhji/dataset/VOCdevkit/VOC2012/Annotations/"

np.random.seed(1)
from backend import parse_annotation
train_image_2012, seen_train_labels_2012 = parse_annotation(train_annot_folder_2012,
                                                  train_image_folder_2012, 
                                                  labels=LABELS)
print("N train_2012 = {}".format(len(train_image_2012)))

### The location where the VOC2012 data is saved.
train_image_folder_2007 = "/home/ali_bhji/dataset/VOCdevkit/VOC2007/JPEGImages/"
train_annot_folder_2007 = "/home/ali_bhji/dataset/VOCdevkit/VOC2007/Annotations/"

np.random.seed(1)
from backend import parse_annotation
train_image_2007, seen_train_labels_2007 = parse_annotation(train_annot_folder_2007,
                                                  train_image_folder_2007, 
                                                  labels=LABELS)
print("N train_2007 = {}".format(len(train_image_2007)))


train_image, seen_train_labels=[],[]
train_image.extend(train_image_2007)
train_image.extend(train_image_2012)

seen_train_labels.extend(seen_train_labels_2007)
seen_train_labels.extend(seen_train_labels_2012)

print("N train VOC2012+2007 = {}".format(len(train_image)))      



from backend import SimpleBatchGenerator

BATCH_SIZE        = 10
IMAGE_H, IMAGE_W  = 416, 416
GRID_H,  GRID_W   = 13 , 13
TRUE_BOX_BUFFER   = 50
BOX               = int(len(ANCHORS)/2)

generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'LABELS'          : LABELS,
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}


def normalize(image):
    return image / 255.
train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                             norm=normalize, shuffle=True)



from backend import define_YOLOv2, set_pretrained_weight, initialize_weight
CLASS             = len(LABELS)
model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, 
                                  trainable=False)
model.summary()


path_to_weight = "./yolov2.weights"
nb_conv        = 22
model          = set_pretrained_weight(model,nb_conv, path_to_weight)
layer          = model.layers[-4] # the last convolutional layer
initialize_weight(layer,sd=1/(GRID_H*GRID_W))


from backend import custom_loss_core 
help(custom_loss_core)


GRID_W             = 13
GRID_H             = 13
BATCH_SIZE         = 34
LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT    = 5.0
LAMBDA_COORD     = 1.0
LAMBDA_CLASS     = 1.0
    
def custom_loss(y_true, y_pred):
    return(custom_loss_core(
                     y_true,
                     y_pred,
                     true_boxes,
                     GRID_W,
                     GRID_H,
                     BATCH_SIZE,
                     ANCHORS,
                     LAMBDA_COORD,
                     LAMBDA_CLASS,
                     LAMBDA_NO_OBJECT, 
                     LAMBDA_OBJECT))
    

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop

dir_log = "logs/"
try:
    os.makedirs(dir_log)
except:
    pass


BATCH_SIZE   = 32
generator_config['BATCH_SIZE'] = BATCH_SIZE

early_stop = EarlyStopping(monitor='loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('weights_yolo_on_voc2012.h5', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)


optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)  

model.fit_generator(generator        = train_batch_generator, 
                    steps_per_epoch  = len(train_batch_generator), 
                    epochs           = 50, 
                    verbose          = 1,
                    #validation_data  = valid_batch,v
                    #validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint], 
                    max_queue_size   = 3)
    

    