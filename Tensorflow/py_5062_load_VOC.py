Image_Folder="JPEGImages"
Label_Folder="Annotations"
IMG_extention='.jpg'

def __init__(self,root,train=True,transform=None,target_transform=None,resize=445, class_path='./voc.name'):
    self.root=root
    self.transform=transform
    self.target_transform=target_transform
    self.train=train
    self.resize_factor=resize
    self.class_path=class_path


    with open(class_path) as f:
        self.classes=f.read().splitlines()

    if not self._check_exists():
        raise RuntimeError("Data set not found.")

    self.data=self.cvtData()

    def _check_exists(self):
        print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))