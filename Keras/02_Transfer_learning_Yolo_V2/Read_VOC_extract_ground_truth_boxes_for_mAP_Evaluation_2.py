import os
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd 

def extract_single_xml_file(tree):
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
        obj=[]
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
        
        obj.append(name)
        obj.append(Xc)
        obj.append(yc)
        obj.append(w)
        obj.append(h)
        obj.append(width)
        obj.append(height)
        obj.append(depth)       
        obj.append(difficult)
        Nobj += 1
        row_list.append(obj)
    row["Nobj"] = Nobj
    # print('\n')
    #row_list---> [class_name xmin ymin xmax ymax width height depth difficult] 
    return(row_list)

Pascal_VOC_2012_root= 'C:\\Users\\alibh\\VOCdevkit\\VOC2012\\'
annotation_foler='Annotations'
images_folder='JPEGImages'

Evaluation_path='.\\Evaluation_mAP\\'
GT_fldr='ground-truth'

if os.path.exists(os.path.join(os.getcwd(),Evaluation_path+GT_fldr)):
    pass
else:
    os.system('mkdir -p '+ os.path.join(os.getcwd(),Evaluation_path+GT_fldr))
    # print('...[Command][mkdir]'+ 'The {} directory is created.'.format(args['output']) )

Pascal_annotation_path=os.path.join(Pascal_VOC_2012_root,annotation_foler)

for i,file in enumerate(os.listdir(Pascal_annotation_path)):
    if not file.startswith('.'): ## do not include hidden folders/files
        tree = ET.parse(os.path.join(Pascal_annotation_path,file))
        rows = extract_single_xml_file(tree)
    print(i , file)
    with open(os.path.join(Evaluation_path+GT_fldr,file.split('.')[0]+'.txt'),'w') as f:
        for row in rows:
            # # scale box position to [0,1]
            # # row is [class_name xmin ymin xmax ymax width height depth difficult] 
            # row[1]=row[1]/row[5]
            # row[3]=row[3]/row[5]
            # row[2]=row[2]/row[6]
            # row[4]=row[4]/row[6]
            # remove difficult images
            
            if len(row)!=9:
                continue
            if row[8]=='1':
                continue
            f.write('{}\n'.format(row))
        

