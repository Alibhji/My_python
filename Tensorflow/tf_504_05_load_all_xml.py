import os
import sys
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

dataset_path = sys.argv[1]

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))

for xml_file in ann_files:
    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    print("Image properties\nwidth : {}\nheight : {}\nchannels : {}\n".format(width, height, channels))

    objects = root.findall("object")
    print("Objects Description")
    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = bndbox.find("xmin").text
        ymin = bndbox.find("ymin").text
        xmax = bndbox.find("xmax").text
        ymax = bndbox.find("ymax").text

        print("class : {}\nxmin : {}\nymin : {}\nxmax : {}\nymax : {}\n".format(name, xmin, ymin, xmax, ymax))

    print("XML parsing END")