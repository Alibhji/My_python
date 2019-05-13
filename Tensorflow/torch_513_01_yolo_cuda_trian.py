from utilities.dataloader import VOC
from imgaug import augmenters as iaa

import torchvision.transforms as transform
from utilities.augmentation import Augmenter

from torch.utils.data import dataloader
import torch
import torch.nn as nn

import yolov1
from yolov1 import detection_loss_4_yolo

from utilities.dataloader import detection_collate
from utilities.utils import save_checkpoint

#print(torch.cuda.is_available())



import os



#data_path = "/home/ali/VOCdevkit/VOC2012"
#class_path = "/home/ali/VOCdevkit/VOC2012/voc.names"
data_path = "/home/ali/VOCdevkit/VOC2012_light"
class_path = "/home/ali/VOCdevkit/VOC2012_light/voc.names"

batch_size = 1
#device = 'cpu'
device = 'cuda'
check_point_path = "/home/ali/PycharmProjects/yolo/out/save_model/"

use_visdom=True

seq = iaa.Sequential([])
compsed = transform.Compose([Augmenter(seq)])

with open(class_path) as f:
    class_list = f.read().splitlines()

train_data = VOC(root=data_path, transform=compsed, class_path=class_path)

train_loader = dataloader.DataLoader(dataset=train_data,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=detection_collate)

# model


dropout = 0.4
num_class = 20
learning_rate = .4
num_epochs = 6

net = yolov1.YOLOv1(params={"dropout": dropout,
                            "num_class": num_class})


if device == 'cpu':
    model = nn.DataParallel(net).cpu()
else:
    model=torch.nn.DataParallel(net).cuda()



optimaizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                              weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimaizer, gamma=0.95)

total_step = len(train_loader)

# print(total_step)

total_train_step = num_epochs * total_step

for epoch in range(1, num_epochs + 1):

    if (epoch == 200) or (epoch == 400):
        scheduler.step()

    for i, (images, labels, sizes) in enumerate(train_loader):
        current_train_step = (epoch - 1) * total_step + i + 1
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # calc Loss
        loss, \
        obj_coord1_loss, \
        obj_size1_loss, \
        obj_class_loss, \
        noobjness1_loss, \
        objness1_loss = detection_loss_4_yolo(outputs, labels, device)
        # print(current_train_step)

        # backward and optimze
        optimaizer.zero_grad()
        loss.backward()
        optimaizer.step()

        print(
            'epoch: [{}/{}], total step:[{}/{}] , batchstep [{}/{}], lr: {},'
            'total_loss: {:.4f}, objness1: {:.4f}, class_loss: {:.4f}'.format(epoch, num_epochs,
                                                                              current_train_step, total_train_step,
                                                                              i + 1, total_step, learning_rate,
                                                                              loss.item(), obj_coord1_loss,
                                                                              obj_size1_loss, obj_class_loss))

    if (epoch % 2 == 0):
        '''
        torch.save({'test': epoch}, 'cc.zip')
        print("Saved...")

        '''
        save_checkpoint({'epoch':epoch+1,
                             'arch': "YOLOv1",
                             'state_dict':model.state_dict(),
                             },False, filename=os.path.join(check_point_path,'ep{:05d}_loss{:.04f}_lr{}.pth.tar'.format(epoch,
                                                            loss.item(),learning_rate,))
                            )
        print("The check point is saved")


