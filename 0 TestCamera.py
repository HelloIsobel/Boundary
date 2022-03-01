import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FPS, 30)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("test", frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.imwrite("test.jpg", frame)
        break
