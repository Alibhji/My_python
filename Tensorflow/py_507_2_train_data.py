
from py_507_VOC_class import VOC



train_dataset = VOC(root='/home/ali/VOCdevkit/VOC2012',
                    transform=transforms.ToTensor(),
                    class_path=class_path)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=detection_collate)
