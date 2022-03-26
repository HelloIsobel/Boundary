import os
import cv2

path = r'D:\Desktop\fuyang_label'
rev_imgpath = 'D:/Desktop/fuyang_label_rev/'
if not os.path.exists(rev_imgpath):
    os.makedirs(rev_imgpath)

for root, dirs, files in os.walk(path):
    for file in files:
        imgpath = os.path.join(root, file)
        img = cv2.imread(imgpath, 0)
        img1 = 255 - img

        cv2.imwrite(rev_imgpath + file, img1)
