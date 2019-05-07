import os
import sys

from PIL import Image
from PIL import ImageDraw

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

dataset_path=sys.argv[1]

image_folder="JPEGImages"
annotation_folder="Annotations"

#print(os.path.join(dataset_path,annotation_folder))
ann_root,ann_dir,ann_files= next(os.walk(os.path.join(dataset_path,annotation_folder)))
img_root,img_dir,img_files=next(os.walk(os.path.join(dataset_path,image_folder)))


for xml_file in ann_files:
    #print(ann_root,"-->",xml_file)
    #print(".".join([xml_file.split(".")[0],"jpg"]))
    image_name=img_files[img_files.index(".".join([xml_file.split(".")[0],"jpg"]))]
    image_file=os.path.join(img_root,image_name)
    image=Image.open(image_file).convert("RGB")
    draw=ImageDraw.Draw(image)
    print(image_file)
    xml=open(os.path.join(ann_root,xml_file),'r')
    tree=ET.parse(xml)
    root=tree.getroot()

    size=root.find("size")

    width=size.find("width").text
    height=size.find("height").text
    channel=size.find("depth").text

    objects=root.findall("object")

    for _object in objects:
        name=_object.find("name").text
        bndbox=_object.find("bndbox")
        xmin=int(bndbox.find("xmin").text)
        ymin=int(bndbox.find("ymin").text)
        xmax=int(bndbox.find("xmax").text)
        ymax=int(bndbox.find("ymax").text)
        box=draw.rectangle(((xmin,ymin),(xmax,ymax)),outline="red")
        draw.text((xmin,ymin),name)

    plt.figure(figsize=(25,20))
    plt.imshow(image)
    plt.show()
    plt.close()













