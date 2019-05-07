import  os
import  sys

dataset_path=sys.argv[1]
image_folder="JPEGImages"
annotations_folder="Annotations"


for root, dirs, files in os.walk(os.path.join(dataset_path,annotations_folder)):
   for name in files:
      print(os.path.join(root, name))
   for name in dirs:
      print(os.path.join(root, name))


'''
ann_root , ann_dir , ann_files=next(os.walk(os.path.join(dataset_path,annotations_folder)))
print("ROOT : {}\n".format(ann_root))
print("DIR : {}\n".format(ann_dir))
print("FILES : {}\n".format(ann_files)) '''