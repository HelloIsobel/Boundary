import numpy as np
import torch
import os
import cv2
import time
import pandas as pd
from model.new_unet_model import U_net

if __name__ == "__main__":
    start1 = time.perf_counter()  # 总时间
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = U_net(1)
    # 将网络拷贝到device中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('resize_1_2_bestmodel.pth', map_location=device))
    # 测试模式
    net.eval()

    # 打开摄像头
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FPS, 30)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    save_res_dirpath = 'data/camera/'
    if not os.path.exists(save_res_dirpath):
        os.mkdir(save_res_dirpath)

    i = 0
    runningtime = []
    runningtime_excelpath = save_res_dirpath+'camera_runningtime.csv'
    while cap.isOpened():
        start = time.perf_counter()
        ret, img_src = cap.read()
        # 改变图片大小
        shape = img_src.shape
        img_src = cv2.resize(img_src, (int(shape[1]/2), int(shape[0]/2)))
        # 转为灰度图
        img = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        end = time.perf_counter()

        runningtime.append(end - start)
        print("the num of image is :", i)
        print("runningtime is :", end - start)

        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        # # # 开运算——先腐蚀再膨胀
        # # 膨胀和腐蚀反着来（因为本来前景是白色，但是我们这里是黑色）
        # kernel = np.ones((5, 5), np.uint8)
        # opening = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)  # 先腐蚀再膨胀 这里对应闭运算

        # 保存图片
        cv2.imwrite(save_res_dirpath+"{}.jpg".format(i), pred)
        i = i + 1

        cv2.imshow("capture", pred)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end1 = time.perf_counter()
    runningtime.append(end1 - start1)
    print("all running time is : ", end1 - start1)

    data = pd.DataFrame(runningtime)
    data.to_csv(runningtime_excelpath)
