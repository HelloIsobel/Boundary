import os
import cv2

fps = 30  # 保存视频的FPS，可以适当调整
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('video.avi', fourcc, fps, (1280, 1024))  # 最后一个是保存图片的尺寸

path = r"C:\Users\617\Desktop\zhuanwan"
for root, dirs, files in os.walk(path):
    for file in files:
        imgpath = os.path.join(root, file)
        frame = cv2.imread(imgpath)
        videoWriter.write(frame)

    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
videoWriter.release()
