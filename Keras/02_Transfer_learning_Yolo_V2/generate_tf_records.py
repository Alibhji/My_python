import tensorflow as tf
import matplotlib.image as mpimg
import os


import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict


pascal_voc_root='/home/ali/VOCdevkit/VOC2012'
image_fldr='JPEGImages'
annotation_foler='Annotations'
tf_writer_name='voc12_tf_record'


voc12_image_dir=os.path.join(pascal_voc_root,image_fldr)
Pascal_annotation_path=os.path.join(pascal_voc_root,annotation_foler)


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
        shape=list(shape)
        # rows=tf.train.Int64List(value=[shape[0]])
        # cols=tf.train.Int64List(value=[shape[1]])
        # channels=tf.train.Int64List(value=[shape[2]])
        shape=tf.train.Int64List(value=shape)
        
        img_str=tf.train.BytesList(value=[img.tostring()])
        
        img_name=img_path.split('/')[-1]
        ann=os.path.join(pascal_voc_root,"{}/{}{}".format(annotation_foler,img_name.split('.')[0],'.xml'))
        
        img_name=tf.train.BytesList(value=[img_path.split('/')[-1].encode('utf-8')])
        
        tree = ET.parse(ann)
        ann=self.extract_single_xml_file(tree)
        objs_class=[objj['class'] for objj in ann]
        xmins=[objj['xmin'] for objj in ann]
        ymins=[objj['ymin'] for objj in ann]
        xmaxs=[objj['xmax'] for objj in ann]
        ymaxs=[objj['ymax'] for objj in ann]
        
        
        # objs_class=tf.train.Int64List(value=[objs_class])
        objs_class = ' '.join(objs_class)
        objs_class=tf.train.BytesList(value=[objs_class.encode('utf-8')])
        
        # print(objs_class)
        xmins= tf.train.FloatList(value=xmins)
        ymins= tf.train.FloatList(value=ymins)
        xmaxs= tf.train.FloatList(value=xmaxs)
        ymaxs= tf.train.FloatList(value=ymaxs)
        # print( aaa)
        # ann=np.array(ann)
        # r=(ann.shape[0])
        # bbx={}
        # for rr in range(r):
        #     bbx['object']=
        
        # objcs,boundboxes=[],[]
        # if(len(ann)):
        #     objcs,boundboxes=ann[:,0],ann[:,1:]
        # else:
        #     boundboxes=np.array([])
        # objcs=' '.join(objcs)
        # objcs=tf.train.BytesList(value=[objcs.encode('utf-8')])
        # boundboxes= tf.train.BytesList(value=[boundboxes.tostring()])
        
        # print(boundboxes)
        # print(' '.join(objcs))
        
        
        # the image annotation should be there
        # 
        features=tf.train.Features( feature={
            'File_name' : tf.train.Feature(bytes_list = img_name),
            # 'rows' : tf.train.Feature(int64_list= rows),
            # 'cols' :tf.train.Feature(int64_list=cols),
            # 'channels' : tf.train.Feature (int64_list=channels),
            'image' : tf.train.Feature(bytes_list= img_str),
            'shape' : tf.train.Feature(int64_list= shape),
            'classes': tf.train.Feature(bytes_list=objs_class),
            'xmin': tf.train.Feature(float_list=xmins),
            'xmax': tf.train.Feature(float_list=xmaxs),
            'ymin': tf.train.Feature(float_list=ymins),
            'ymax': tf.train.Feature(float_list=ymaxs)
             
            }
        )
        # print(img,rows,img_name)
        # print(tf.train.Example(features=features))
        return tf.train.Example(features=features)
    
    
    def extract_single_xml_file(self,tree):
        Nobj = 0
        row_list=[]
        row  = OrderedDict()
        size=[]
        size = tree.find("size")
        depth =int(size.find('depth').text)
        height=int(size.find('height').text)
        width=int(size.find('width').text)
        objects=tree.findall("object")
        for elem in objects:
            obj={}
            # obj=[]
            name=(elem.find('name').text)
            difficult=int(elem.find('difficult').text)
            bndbox=elem.find('bndbox')
            xmin=float(bndbox.find('xmin').text)
            ymin=float(bndbox.find('ymin').text)
            xmax=float(bndbox.find('xmax').text)
            ymax=float(bndbox.find('ymax').text)
            
            Xc=xmin+xmax/2.0
            yc=ymin+ymax/2.0
            w=xmax-xmin
            h=ymax-ymin 

            if difficult ==1:
                continue
            obj['class'],obj['xmin'],obj['xmax'],obj['ymin'],obj['ymax']=name , xmin,xmax,ymin,ymax
            
            # obj.append(name)
            # obj.append(Xc)
            # obj.append(yc)
            # obj.append(w)
            # obj.append(h)

            # obj.append(name)
            
            # obj.append(xmin)
            # obj.append(ymin)
            # obj.append(xmax)
            # obj.append(ymax)
            
            

            # obj.append(width)
            # obj.append(height)
            # obj.append(depth)       
            # obj.append(difficult)
            Nobj += 1
            row_list.append(obj)
        row["Nobj"] = Nobj
        # print('\n')
        #row_list---> [class_name xmin ymin xmax ymax width height depth difficult] 
        return(row_list)
        
        
        
        
a=generate_tf_records('a')
        