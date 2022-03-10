import os
import cv2

# 打开摄像头
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FPS, 30)
img_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(img_size)

fps = 30  # 保存视频的FPS，可以适当调整
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('saveVideo.avi', fourcc, fps, img_size)  # 最后一个是保存图片的尺寸

while cap.isOpened():
    ret, frame = cap.read()
    pred = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    pred[pred >= 100] = 255
    pred[pred < 100] = 0
    frame = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    videoWriter.write(frame)

    cv2.imshow("capture", pred)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
videoWriter.release()
