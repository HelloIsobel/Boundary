import numpy as np
import torch
import os
import cv2
import time
import pandas as pd
import argparse
from model.new_unet_model import U_net


def pred():
    start1 = time.perf_counter()  # 总时间
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = U_net(1)
    # 将网络拷贝到device中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load(args.bestmodel, map_location=device))  # TODO:'resize_1_2_bestmodel.pth'
    # 测试模式
    net.eval()

    if args.capOpen:
        # 打开摄像头：保证输入时是大角度的视野
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(args.open_video)  # TODO：修改video名

    if not os.path.exists(args.save_pred):
        os.makedirs(args.save_pred)

    i = 0
    runningtime = []
    runningtime_excelpath = args.save_pred + 'camera_runningtime.csv'
    while cap.isOpened():
        start = time.perf_counter()
        ret, img_src = cap.read()
        # 改变图片大小，双线性插值让图像变小
        shape = img_src.shape
        img_src = cv2.resize(img_src, (int(shape[1] / 2), int(shape[0] / 2)))
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

        frame = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
        # 保存图片
        cv2.imwrite(args.save_pred + "{}.jpg".format(i), frame)

        i = i + 1
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end1 = time.perf_counter()
    runningtime.append(end1 - start1)
    print("all running time is : ", end1 - start1)

    data = pd.DataFrame(runningtime)
    data.to_csv(runningtime_excelpath)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    parser = argparse.ArgumentParser(description="Segment farmland area and non-farmland area.")
    parser.add_argument("--bestmodel", type=str, metavar="", default="resize_1_2_bestmodel.pth",
                        help="Save a bestmodel")
    parser.add_argument("--save_pred", metavar="", type=str, default="data/" + time_str + "/camera/",
                        help="Save pred images")
    parser.add_argument("--save_video", metavar="", type=str, default="data/" + time_str + "/video.avi",
                        help="Save video")
    parser.add_argument("--capOpen", metavar="", type=str, default=False, help="open cap")
    # parser.add_argument("--videoOpen", metavar="", type=str, default=True, help="open video")
    parser.add_argument("--open_video", metavar="", type=str, default="data/video/yk01.avi", help="Open video")

    args = parser.parse_args()
    print(args)
    pred()

    # 没有开摄像头 all running time is :  1.2212485
