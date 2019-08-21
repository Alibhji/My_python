# Ali Babolhavaeji- 8/20/2019
# Yolo V2 - Keras_ transfer learning - python 3.5 - tesnsorflow-gpu - keras-gpu, cuda 10

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from tensorflow.python.client import device_lib
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())



def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):

    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    filtering_mask = box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask) 
    return scores, boxes, classes

with tf.Session() as test_a:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))

def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou
# GRADED FUNCTION: yolo_non_max_suppression

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):  
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    nms_indices = tf.image.non_max_suppression( boxes, scores, max_boxes_tensor, iou_threshold)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes

with tf.Session() as test_b:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=100, score_threshold=0.3, iou_threshold=0.3):    
    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)
    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
    ### END CODE HERE ###
    
    return scores, boxes, classes

with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.) 


yolo_model = load_model("..\\..\\..\\Model\\yolo.h5")
# yolo_model.summary()
yolo_model.save('..\\..\\..\\Model\\yolo_model.h5')
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
scores, boxes, classes = yolo_eval(yolo_outputs, input_image_shape)

Pascal_VOC_2012_Path= 'C:\\Users\\alibh\\VOCdevkit'

import numpy as np
import cv2
from PIL import Image
from io import BytesIO

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    # cv2.imshow('gray',gray)
    img = Image.fromarray(frame)
    new_width  = 1280
    new_height = 720
    img = img.resize((new_width, new_height), Image.BICUBIC)
    frame = np.array(img)
    model_image_size = (608, 608)
    resized_image = img.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    b = BytesIO()
    img.save(b,format="jpeg")
    image=Image.open(b)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data,
                                                                                input_image_shape: [image.size[1], image.size[0]],
                                                                                K.learning_phase(): 0})
    
    colors = generate_colors(class_names)
    # draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        label = '{} {:.2f} {} {} {} {}'.format(predicted_class, score, left, top, right, bottom)
        print(label)
        
        
    # image = np.array(image)
    # cv2.imshow('RGB image',image)
    # cv2.imshow('RGB image',frame)
     

    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# import freenect
# import cv2
# import numpy as np
# from io import BytesIO  
 
# #function to get RGB image from kinect
# def get_video():
#     array,_ = freenect.sync_get_video()
#     array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
#     return array
 
# #function to get depth image from kinect
# def get_depth():
#     array,_ = freenect.sync_get_depth()
#     array = array.astype(np.uint8)
#     return array
 
# if __name__ == "__main__":
#     while 1:
#         #get a frame from RGB camera
#         frame = get_video()
#         img = PIL.Image.fromarray(frame)
#         new_width  = 1280
#         new_height = 720
#         img = img.resize((new_width, new_height), PIL.Image.BICUBIC)
#         frame = np.array(img)
#         model_image_size = (608, 608)
#         resized_image = img.resize(tuple(reversed(model_image_size)), PIL.Image.BICUBIC)
#         image_data = np.array(resized_image, dtype='float32')
#         image_data /= 255.
#         image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
#         b = BytesIO()
#         img.save(b,format="jpeg")
#         image=PIL.Image.open(b)
#         out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data,
#                                                                                        input_image_shape: [image.size[1], image.size[0]],
#                                                                                        K.learning_phase(): 0})

#         colors = generate_colors(class_names)
#         draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#         image = np.array(image)
#         cv2.imshow('RGB image',image)
#         # cv2.imshow('RGB image',frame)
#         k = cv2.waitKey(5) & 0xFF
#         if k == 27:
#             break
#     cv2.destroyAllWindows()