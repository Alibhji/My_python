

import tensorflow as tf 
filename= '../../../Model/voc12_tf_record'

dataset = tf.data.TFRecordDataset(filename)

def read_tfrecord(example):
    
    features={
    'File_name' : tf.io.FixedLenFeature([], tf.string),
    # 'rows' : tf.train.Feature(int64_list= rows),
    # 'cols' :tf.train.Feature(int64_list=cols),
    # 'channels' : tf.train.Feature (int64_list=channels),
    'image'  : tf.io.FixedLenFeature([],  tf.string),
    'shape'  : tf.io.FixedLenFeature([],  tf.int32),
    'classes': tf.io.FixedLenFeature([],  tf.string),
    'xmin'   : tf.io.FixedLenFeature([],  tf.float32),
    'xmax'   : tf.io.FixedLenFeature([],  tf.float32),
    'ymin'   : tf.io.FixedLenFeature([],  tf.float32),
    'ymax'   : tf.io.FixedLenFeature([],  tf.float32),
        
    }
    
    
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    
    return image




