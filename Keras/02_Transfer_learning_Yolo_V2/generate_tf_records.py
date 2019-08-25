import tensorflow as tf
import matplotlib.image as mpimg
import os

pascal_voc_root='/home/ali/VOCdevkit/VOC2012'
image_fldr='JPEGImages'
tf_writer_name='voc12_tf_record'

voc12_image_dir=os.path.join(pascal_voc_root,image_fldr)


class generate_tf_records:
    def __init__(self, labels):
        self.labels=labels
        self.imgs_voc12=self.get_image_absoulte_path(voc12_image_dir)
        # print('[in the func-1] :',self.imgs_voc12)
        ex_obj=self.genrate_tf_exapmle_obj_from_img(self.imgs_voc12[1])
        # print(ex_obj)
        with tf.python_io.TFRecordWriter (tf_writer_name) as writer:
            for i,img in enumerate(self.imgs_voc12) :
                example_obj= self.genrate_tf_exapmle_obj_from_img(img)
                writer.write(example_obj.SerializeToString())
                print('image {:05d}/ {}  --> {}'.format(i,len(self.imgs_voc12),img.split('/')[-1]))
        
        
    def get_image_absoulte_path(self, dataset_path):
        imgs_path=[os.path.join(dataset_path,i) for i in os.listdir(dataset_path) ]
        return imgs_path
        
    def genrate_tf_exapmle_obj_from_img(self, img_path):
        img=mpimg.imread(img_path)
        shape=img.shape
        rows=tf.train.Int64List(value=[shape[0]])
        cols=tf.train.Int64List(value=[shape[1]])
        channels=tf.train.Int64List(value=[shape[2]])
        img_str=tf.train.BytesList(value=[img.tostring()])
        img_name=img_path.split('/')[-1]
        img_name=tf.train.BytesList(value=[img_path.split('/')[-1].encode('utf-8')])
        # the image annotation should be there
        # 
        features=tf.train.Features( feature={
            'File_name' : tf.train.Feature(bytes_list = img_name),
            'rows' : tf.train.Feature(int64_list= rows),
            'cols' :tf.train.Feature(int64_list=cols),
            'channels' : tf.train.Feature (int64_list=channels),
            'image' : tf.train.Feature(bytes_list= img_str)
            }
        )
        # print(img,rows,img_name)
        # print(tf.train.Example(features=features))
        return tf.train.Example(features=features)
        
        
        
        
a=generate_tf_records('a')
        