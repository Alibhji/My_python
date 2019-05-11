import sys
import os
import torch




"""

train_dataset = VOC(root='/home/ali/VOCdevkit/VOC2012',
                    transform=transforms.ToTensor(),
                    class_path=class_path)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=detection_collate)



class VOC(data.Dataset):

    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, root, train=True, transform=None, target_transform=None, resize=448, class_path='./voc.names'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.class_path = class_path

        with open(class_path) as f:
            self.classes = f.read().splitlines()

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data = self.cvtData()

    def _check_exists(self):
        print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def cvtData(self):

        result = []
        voc = cvtVOC()

        yolo = cvtYOLO(os.path.abspath(self.class_path))
        flag, self.dict_data =voc.parse(os.path.join(self.root, self.LABEL_FOLDER))

        try:

            if flag:
                flag, data =yolo.generate(self.dict_data)

                keys = list(data.keys())
                keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

                for key in keys:
                    contents = list(filter(None, data[key].split("\n")))
                    target = []
                    for i in range(len(contents)):
                        tmp = contents[i]
                        tmp = tmp.split(" ")
                        for j in range(len(tmp)):
                            tmp[j] = float(tmp[j])
                        target.append(tmp)

                    result.append({os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])) : target})

                return result

        except Exception as e:
            raise RuntimeError("Error : {}".format(e))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        key = list(self.data[index].keys())[0]

        img = Image.open(key).convert('RGB')
        current_shape = img.size
        img = img.resize((self.resize_factor, self.resize_factor))

        target = self.data[index][key]

        if self.transform is not None:
            img, aug_target = self.transform([img, target])
            img = torchvision.transforms.ToTensor()(img)

        if self.target_transform is not None:
            # Future works
            pass

        return img, aug_target, current_shape

"""