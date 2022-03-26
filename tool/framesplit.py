
"""
提取每一帧
"""
import cv2
import os

cap = cv2.VideoCapture(r"D:\Project\zxy\Boundary\video_20220323_1711.avi")
i = 0

name_dir = r"D:\Desktop\fuyang_frame_10"
if not os.path.exists(name_dir):
    os.makedirs(name_dir)

while cap.isOpened():
    rval, frame = cap.read()
    if rval == True:
        i += 1
        if i % 10 == 0:
            cv2.imwrite(name_dir + '/{}.jpg'.format(int(i / 50)), frame)
    else:
        break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()



