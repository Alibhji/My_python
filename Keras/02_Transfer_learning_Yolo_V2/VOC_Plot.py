
import cv2
import os

Pascal_VOC_2012_root= '/home/ali/VOCdevkit/VOC2012/'
annotation_foler='Annotations'
images_folder='JPEGImages'
file="2008_001054"
# # Windows
# Evaluation_path='.\\Evaluation_mAP\\'
Evaluation_path='./Evaluation_mAP/'
# Linux-Ubuntu
Yolo_Estimation_output_fldr='Yolo_estimation_output'
GT_fldr='ground-truth'


for i,file in enumerate(os.listdir(os.path.join(Evaluation_path,Yolo_Estimation_output_fldr))):
    file=file.split('/')
    file=file[-1].split('.')[0]
    print(1,file)
    img1=cv2.imread(os.path.join(Pascal_VOC_2012_root+images_folder,file)+'.jpg',cv2.IMREAD_COLOR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    with open(os.path.join(Evaluation_path+Yolo_Estimation_output_fldr,file)+'.txt') as f:
        rows=f.readlines()
        for row in rows:
            row=row.split('\n')
            row=row[0].split(' ')
            print(row)
            center_x=int(float(row[2])+float(row[4]))/2.0
            center_y=int(float(row[3])+float(row[5]))/2.0
            cv2.putText(img1, ''.join(row[0]+ ': '+row[1]),(int(center_x),int(center_y)), font, .7, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(img1,(int(row[2]),int(row[3])),(int(row[4]),int(row[5])),(0,0,255),1)

    with open(os.path.join(Evaluation_path+GT_fldr,file)+'.txt') as f:
        rows=f.readlines()
        for row in rows:
            row=row.split('\n')
            row=row[0].split(' ')
            row[1:]=[float(i) for i in row[1:]]
            print(row)
            center_x=((row[1])+(row[2]))/2.0
            center_y=((row[2])+(row[4]))/2.0
            cv2.putText(img1, ''.join(row[0]),(int(center_x),int(center_y)), font, .7, (255,255,255), 1, cv2.LINE_AA)
            cv2.rectangle(img1,(int(row[1]),int(row[2])),(int(row[3]),int(row[4])),(255,255,255),1)

    print(os.path.join(Pascal_VOC_2012_root+images_folder,file)+'.jpg')
    # print(img1)

    # cv2.line(img1,(0,0),(150,150),(255,255,255),10)
    # cv2.rectangle(img1,(10,10),(150,150),(255,255,255),10)



    cv2.imshow('image',img1)
    cv2.waitKey(5000)

cv2.destroyAllWindows()
