import numpy as np
import torch
import cv2
import math
import os
import time
import argparse
import pandas as pd
from model.new_unet_model import U_net
from tool.functions import fit_line_by_ransac
from tool.functions import get_maxContour


def DecectBoundary():
    global videoWriter
    start0 = time.perf_counter()  # 总时间
    # ***************** 神经网络搭建 ***************** #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = U_net(1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load(args.bestmodel, map_location=device))
    # 测试模式
    net.eval()

    # ***************** 读取视频路径和写入视频路径 ***************** #
    if args.capOpen:
        # 打开摄像头：保证输入时是大角度的视野
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FPS, 30)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))
    else:
        cap = cv2.VideoCapture(args.open_video)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))

    # # ***************** 保存视频 ***************** #
    # 帧率
    if args.saveVideoOpen:
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(args.save_video, fourcc, fps, size)

    # ***************** 初始化 ***************** #
    flag, flag2 = 0, 0  # 判断是否使用扩展检测边界线法
    left_flag = 0  # 判断农机是否从左边驶入
    a, b, a2, b2 = 0, 0, 0, 0  # 检测直线的斜率和截距

    frame_dir = args.save_dirpath + "frame/"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    candidateBoundary_dir = args.save_dirpath + "candidateBoundary/"
    if not os.path.exists(candidateBoundary_dir):
        os.makedirs(candidateBoundary_dir)

    Boundary_dir = args.save_dirpath + "Boundary/"
    if not os.path.exists(Boundary_dir):
        os.makedirs(Boundary_dir)

    runningtime = []
    runningtime_excelpath = args.save_dirpath + 'runningtime.csv'

    i = 0  # 每一帧命名
    # ***************** 帧处理 ***************** #
    while cap.isOpened():
        rval, frame = cap.read()
        start = time.perf_counter()
        # 转为灰度图
        img_resize = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        w = img_resize.shape[1]

        img = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        # ******************************** 画图，预测红色透明面积【可注释】 ******************************* #
        index = np.argwhere(pred == 0)
        for j in range(len(index)):
            x = index[j, 0]
            y = index[j, 1]

            # # ** red
            if img_resize[x, y, 2] > 205:
                img_resize[x, y, 2] = 255
            else:
                img_resize[x, y, 2] = img_resize[x, y, 2] + 50

            img_resize[x, y, 0] = img_resize[x, y, 0] / 1.5
            img_resize[x, y, 1] = img_resize[x, y, 1] / 1.5

        # ******************************** 开运算——先腐蚀再膨胀  ******************************* #
        # 膨胀和腐蚀反着来（因为本来前景是白色，但是我们这里是黑色）
        kernel = np.ones((2, 2), np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)  # 先腐蚀再膨胀 这里对应闭运算
        max_contour_edge = get_maxContour(pred, args.T_contour_area)  # 最大轮廓

        # ******************************** 画图，候选边界线【可注释】 ******************************* #
        # for coor in max_contour_edge:
        #     cv2.circle(img_resize, (int(coor[0]), int(coor[1])), 1, (0, 255, 0), 4)  # ( 0, 255, 255)  (255, 0, 0)

        # # 保存地址
        # cv2.imwrite(frame_dir + '{}.jpg'.format(i), frame)
        # cv2.imwrite(candidateBoundary_dir + '{}.jpg'.format(i), img_resize)

        # ******************************** 检测边界线 ******************************* #
        edge_index = np.array([[w + 10, 0]])
        merge_G = cv2.GaussianBlur(pred, (3, 3), 0)
        merge_G = merge_G.astype("uint8")
        edges = cv2.Canny(merge_G, 10, 50, apertureSize=3)  # 提取图像边缘

        # ****************** 判断直线1 采用直接拟合法，还是扩展法 ****************** #
        extend_h = int(args.extend_w / math.sin(abs(math.atan(a) - math.pi / 2)))
        if flag == 1:
            # ****************** 扩展法 ****************** #
            edge_index = np.argwhere(edges == 255)
            edge_index[:, [0, 1]] = edge_index[:, [1, 0]]  # edge_index[:, 0]代表x edge_index[:, 1]代表y

            if len(edge_index) != 0:
                y1 = edge_index[:, 0] * a + b - extend_h
                edge_index = edge_index[y1 < edge_index[:, 1]]
                y2 = edge_index[:, 0] * a + b + extend_h
                edge_index = edge_index[edge_index[:, 1] < y2]
                if len(edge_index) != 0:
                    # 画出前个检测直线的扩展区间
                    cv2.line(img_resize, (0, int(b) - extend_h), (w, int(w * a + b - extend_h)), (0, 255, 0), 3)
                    cv2.line(img_resize, (0, int(b) + extend_h), (w, int(w * a + b + extend_h)), (0, 255, 0), 3)

                    a, b = fit_line_by_ransac(edge_index, sigma=3)  # TODO: sigma
                    cv2.line(img_resize, (0, int(b)), (w, int(w * a + b)), (255, 0, 0), 3)
                else:
                    flag = 0
            else:
                flag = 0
        else:
            # ****************** 直接法 ****************** #
            if len(max_contour_edge) != 0:
                a, b = fit_line_by_ransac(max_contour_edge, sigma=3)  # TODO: sigma
                cv2.line(img_resize, (0, int(b)), (w, int(w * a + b)), (255, 0, 0), 3)
                flag = 1
                # 判断是否从左边进去农田
                centre = np.average(max_contour_edge, axis=0)
                if centre[0] < w / 2:
                    left_flag = 1
                    print("agricultural machine drove in from the Left!")
                else:
                    print("agricultural machine drove in from the Right!")

        # ****************** 判断直线2 采用直接拟合法，还是扩展法 ****************** #
        # TODO： 这里需要添加判断一开始是左边还是右边
        if len(max_contour_edge) != 0:
            if left_flag:
                max_contour_edge2 = max_contour_edge[max_contour_edge[:, 0] > max(edge_index[:, 0])]  # edge2
            else:
                max_contour_edge2 = max_contour_edge[max_contour_edge[:, 0] < min(edge_index[:, 0])]  # edge2

            for coor in edge_index:
                cv2.circle(img_resize, (int(coor[0]), int(coor[1])), 1, (0, 0, 255), 2)  # ( 0, 255, 255)  (255, 0, 0)
            for coor in max_contour_edge2:
                cv2.circle(img_resize, (int(coor[0]), int(coor[1])), 1, (128, 0, 128), 2)  # ( 0, 255, 255)  (255, 0, 0)

            if len(max_contour_edge2):
                area = sum(max_contour_edge2[:, 1])
                print("{}的顶端面积是{}".format(i, area))
                if flag2 == 1:
                    if area > args.T_area:
                        extend_h2 = int(args.extend_w / math.sin(abs(math.atan(a2) - math.pi / 2)))  # 扩展宽度

                        edge_index = np.argwhere(edges == 255)
                        edge_index[:, [0, 1]] = edge_index[:, [1, 0]]

                        if len(edge_index) != 0:
                            y1_2 = edge_index[:, 0] * a2 + b2 - extend_h2
                            edge_index = edge_index[y1_2 < edge_index[:, 1]]
                            y2_2 = edge_index[:, 0] * a2 + b2 + extend_h2
                            edge_index = edge_index[edge_index[:, 1] < y2_2]
                            if len(edge_index) != 0:
                                # 画出前个检测直线的扩展区间
                                cv2.line(img_resize, (0, int(b2) - extend_h2), (w, int(w * a2 + b2 - extend_h2)),
                                         (0, 255, 0), 3)
                                cv2.line(img_resize, (0, int(b2) + extend_h2), (w, int(w * a2 + b2 + extend_h2)),
                                         (0, 255, 0), 3)
                                a2, b2 = fit_line_by_ransac(edge_index, sigma=3)  # TODO: sigma
                                cv2.line(img_resize, (0, int(b2)), (w, int(w * a2 + b2)), (255, 0, 0), 3)
                        else:
                            flag2 = 0
                    else:
                        flag2 = 0
                else:
                    if area > args.T_area:
                        a2, b2 = fit_line_by_ransac(max_contour_edge2, sigma=3)  # TODO: sigma
                        cv2.line(img_resize, (0, int(b2)), (w, int(w * a2 + b2)), (255, 0, 0), 3)
                        flag2 = 1

        end = time.perf_counter()
        runningtime.append(end - start)
        print("the runningtime of [ {} ] image is: {}".format(i, end - start))

        # ********** 作图 ********** #
        cv2.imwrite(Boundary_dir + '{}.jpg'.format(i), img_resize)

        if args.saveVideoOpen:
            img_try = cv2.imread(Boundary_dir + '{}.jpg'.format(i))
            videoWriter.write(img_try)

        i = i + 1

        cv2.imshow("capture", img_resize)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    end0 = time.perf_counter()
    runningtime.append(end0 - start0)
    print("all running time is : ", end0 - start0)

    data = pd.DataFrame(runningtime)
    data.to_csv(runningtime_excelpath)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    parser = argparse.ArgumentParser(description="Segment farmland area and non-farmland area.")
    parser.add_argument("--bestmodel", type=str, metavar="", default="data/376_bestmodel.pth",
                        help="Save a bestmodel")
    parser.add_argument("--T_contour_area", type=int, metavar="", default=10000, help="二值分割后，轮廓面积最小阈值")
    parser.add_argument("--T_area", type=int, metavar="", default=10000, help="第二条线，顶端田埂面积阈值")
    parser.add_argument("--extend_w", type=int, metavar="", default=50, help="扩展宽度")
    parser.add_argument("--save_dirpath", type=str, metavar="", default="data/result/" + time_str + "/",
                        help="Save images")
    # TODO:决定保存视频还是不保存
    parser.add_argument("--saveVideoOpen", metavar="", type=str, default=False, help="save video open")
    parser.add_argument("--save_video", metavar="", type=str, default="data/result/" + time_str + "/boundaryVideo.avi",
                        help="Save video")
    # TODO:决定用相机还是已有视频
    parser.add_argument("--capOpen", metavar="", type=str, default=False, help="open cap")
    parser.add_argument("--open_video", metavar="", type=str, default="data/video/video_20220323_1733_right.avi",
                        help="Open video and print path of video")

    args = parser.parse_args()
    print(args)
    DecectBoundary()

    # 没有开摄像头 all running time is :  1.155508
