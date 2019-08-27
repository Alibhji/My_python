

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import  keras.datasets.

tf_writer_name='../../../Model/voc12_tf.record'
dataset = tf.data.TFRecordDataset(tf_writer_name)

def decode(serialized_example):
    """
    Parses an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

    # 1. define a parser
    features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
        'File_name': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),       
        # 'shape': tf.FixedLenFeature([], tf.int64),
        'classes' : tf.FixedLenFeature([], tf.string),
        'xmin': tf.FixedLenFeature([], tf.float32),
        'xmax': tf.FixedLenFeature([], tf.float32),
        'ymin': tf.FixedLenFeature([], tf.float32),
        'ymax': tf.FixedLenFeature([], tf.float32),
        'rows': tf.FixedLenFeature([], tf.int64),
        'cols': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64)
        
    })
    # 2. Convert the data
    image = tf.decode_raw(features['image'], tf.uint8)
    File_name = features['File_name']
    classes = features['classes']
    xmin = features['xmin']
    ymin = features['ymin']
    xmax = features['xmax']
    ymax = features['ymax']
    rows = features['rows']
    cols = features['cols']
    channels = features['channels']
    

    
    # image = np.reshape(image,(rows,cols,channels))
    image = tf.reshape(image, [rows, cols,channels])
    print('----------------------------------------->',type(image))
    return image, (File_name,classes,xmin,ymin,xmax,ymax,rows,cols,channels)

dataset = dataset.map(decode)
iterator = dataset.make_one_shot_iterator()
image_batch, label_batch = iterator.get_next()

ds= dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=500))
ds.batch(10)

sess = tf.Session()

image_batch, label_batch = sess.run([image_batch, label_batch])
print(image_batch.shape)
print(label_batch)

cv2.imshow('image',image_batch)
cv2.waitKey(500)

cv2.destroyAllWindows()

# imgplot = plt.imshow(image_batch)

