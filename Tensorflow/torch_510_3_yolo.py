from utilities.dataloader import VOC
from imgaug import  augmenters as iaa

import torchvision.transforms as transform
from utilities.augmentation import Augmenter


data_path="/home/ali/VOCdevkit/VOC2012"
class_path="/home/ali/VOCdevkit/VOC2012/voc.names"
seq=iaa.Sequential([])
compsed=transform.Compose([Augmenter(seq)])
train_data=VOC(root=data_path , transform=compsed, class_path=class_path)

