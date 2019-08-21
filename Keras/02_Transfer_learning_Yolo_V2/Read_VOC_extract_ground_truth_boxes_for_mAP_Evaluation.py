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
    for elems in tree.iter():

        if elems.tag == "size":
            size=[]
            for elem in elems:
                row[elem.tag] = int(elem.text)
                size.append(int(elem.text))
        if elems.tag == "object":
            obj=[]
            for elem in elems:               
                if elem.tag == "name":
                    obj.insert(0,str(elem.text))
                    row["bbx_{}_{}".format(Nobj,elem.tag)] = str(elem.text)   
                if elem.tag == "difficult":
                    diff=elem.text              
                if elem.tag == "bndbox":
                    for k in elem:
                        row["bbx_{}_{}".format(Nobj,k.tag)] = float(k.text)
                        obj.append(float(k.text))
                    Nobj += 1
                    obj.extend(size)
                    obj.append(diff)
                    # print('.')  
                    xmin=obj[1]
                    ymin=obj[2]
                    xmax=obj[3]
                    ymax=obj[4]
                    
                    Xc=xmin+xmax/2.0
                    yc=ymin+ymax/2.0
                    w=xmax-xmin
                    h=ymax-ymin 
                      
                    
                    obj[1]=Xc
                    obj[2]=yc
                    obj[3]=w
                    obj[4]=h
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

for file in os.listdir(Pascal_annotation_path):
    if not file.startswith('.'): ## do not include hidden folders/files
        tree = ET.parse(os.path.join(Pascal_annotation_path,file))
        rows = extract_single_xml_file(tree)
    print(rows)
    with open(os.path.join(Evaluation_path+GT_fldr,file.split('.')[0]+'.txt'),'w') as f:
        for row in rows:
            # scale box position to [0,1]
            # row is [class_name xmin ymin xmax ymax width height depth difficult] 
            row[1]=row[1]/row[5]
            row[3]=row[3]/row[5]
            row[2]=row[2]/row[6]
            row[4]=row[4]/row[6]
            # remove difficult images
            if row[8]=='1':
                continue
            f.write('{}\n'.format(row))
        

