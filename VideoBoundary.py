import numpy as np
import torch
import cv2
import random
import math
import os
import time
from model.new_unet_model import U_net


def fit_line_by_ransac(point_list, sigma, iters=50, P=0.99):
    # 使用RANSAC算法拟合直线
    # 迭代最大次数 iters = 1000
    # 数据和模型之间可接受的差值 sigma
    # 希望的得到正确模型的概率P = 0.99

    # 最好模型的参数估计
    best_a = 0  # 直线斜率
    best_b = 0  # 直线截距
    n_total = 0  # 内点数目
    for j in range(iters):
        # 随机选两个点去求解模型
        sample_index = random.sample(range(len(point_list)), 2)
        x_1 = point_list[sample_index[0]][0]
        y_1 = point_list[sample_index[0]][1]
        x_2 = point_list[sample_index[1]][0]
        y_2 = point_list[sample_index[1]][1]
        if x_2 == x_1:
            continue

        # y = ax + b 求解出a，b
        a_ = (y_2 - y_1) / (x_2 - x_1)
        b_ = y_1 - a_ * x_1

        # 算出内点数目
        total_inlier = 0
        for index_ in range(len(point_list)):
            y_estimate = a_ * point_list[index_][0] + b_
            if abs(y_estimate - point_list[index_][1]) < sigma:
                total_inlier += 1

        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > n_total:
            iters = math.log(1 - P) / math.log(1 - pow(total_inlier / len(point_list), 2))
            n_total = total_inlier
            best_a = a_
            best_b = b_

        # 判断是否当前模型已经符合超过一半的点
        if total_inlier > len(point_list) // 2:
            break

    return best_a, best_b


def get_maxContour(bw):
    """得到最大面积连通域"""
    pred_flip = 255 - bw  # 因为下面检测contours默认是包围白色像素点，也就是把黑色像素点当做背景
    pred_flip = pred_flip.astype("uint8")
    contours, hierarchy = cv2.findContours(pred_flip, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    global T_contour_area
    contours_area = []
    max_edge_no_boundary = np.array([])

    if len(contours) != 0:
        for j in range(len(contours)):
            contours_area.append(cv2.contourArea(contours[j]))
        max_area_index = np.argmax(np.array(contours_area))
        # ### 判断阈值 T_contour_area 设置多少合适
        # maxarea = contours_area[int(max_area_index)]
        # print("最大轮廓面积是{}".format(maxarea))

        if contours_area[int(max_area_index)] > T_contour_area:  # TODO： 尺寸未减半时是30000
            max_edge = np.squeeze(contours[max_area_index])
            index = np.argwhere(max_edge[:, 1] == 0)
            index = np.append(index, np.argwhere(max_edge[:, 0] == 0))
            index = np.append(index, np.argwhere(max_edge[:, 0] == bw.shape[1] - 1))
            index = np.append(index, np.argwhere(max_edge[:, 1] == bw.shape[0] - 1))
            index = np.append(np.sort(index), [len(max_edge) - 1])
            index_sub = np.array([0])
            index_sub = np.append(index_sub, index[0:-1])  # 这里是倒数第二个
            sub = index - index_sub
            max_index = np.argwhere(sub == max(sub))
            max_edge_no_boundary = max_edge[int(index[max_index - 1]):int(index[max_index]), :]

    return max_edge_no_boundary


if __name__ == "__main__":
    # ***************** 读取视频路径和写入视频路径 ***************** #
    start = time.clock()
    cap = cv2.VideoCapture("data/video/video2.avi")  # TODO：修改video名

    # ***************** 神经网络搭建 ***************** #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = U_net(1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('resize_1_2_bestmodel.pth', map_location=device))
    # 测试模式
    net.eval()

    # ***************** 帧处理 ***************** #
    flag, flag2 = 0, 0  # 判断是否使用扩展检测边界线法
    a, b, a2, b2 = 0, 0, 0, 0  # 检测直线的斜率和截距

    T_contour_area = 10000  # 二值分割后，轮廓面积最小阈值 # TODO
    T_area = 15000  # 第二条线，顶端田埂面积阈值 # TODO

    i = 0  # 每一帧命名

    # ***************** 帧处理 ***************** #
    while cap.isOpened():
        rval, frame = cap.read()
        if rval:
            # 转为灰度图
            img_resize = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
            w = img_resize.shape[1]
            h = img_resize.shape[0]

            img = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
            # 转为batch为1，通道为1，大小为512*512的数组
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
            # 膨胀和腐蚀反着来（因为本来前景是白色，但是我们这里是黑色）】
            kernel = np.ones((2, 2), np.uint8)
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)  # 先腐蚀再膨胀 这里对应闭运算
            max_contour_edge = get_maxContour(pred)  # 最大轮廓

            # ******************************** 画图，候选边界线【可注释】 ******************************* #
            for coor in max_contour_edge:
                cv2.circle(img_resize, (int(coor[0]), int(coor[1])), 1, (0, 255, 0), 4)  # ( 0, 255, 255)  (255, 0, 0)

            # # 保存地址
            # name_dir = 'result/candidateBoundary'
            # if not os.path.exists(name_dir):
            #     os.mkdir(name_dir)
            # cv2.imwrite(name_dir + '/{}.jpg'.format(i), img_resize)

            # ******************************** 检测边界线 ******************************* #
            edge_index = np.array([[w + 10, 0]])
            merge_G = cv2.GaussianBlur(pred, (3, 3), 0)
            merge_G = merge_G.astype("uint8")
            edges = cv2.Canny(merge_G, 10, 50, apertureSize=3)  # 提取图像边缘

            # ****************** 判断直线1 采用直接拟合法，还是扩展法 ****************** #
            extend_w = 50  # TODO: 扩展宽度
            extend_h = int(extend_w / math.sin(abs(math.atan(a) - math.pi / 2)))
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
                        # cv2.line(img, (0, int(b) - extend_h), (w, int(w * a + b - extend_h)), (0, 255, 0), 3)
                        # cv2.line(img, (0, int(b) + extend_h), (w, int(w * a + b + extend_h)), (0, 255, 0), 3)

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

            # ****************** 判断直线2 采用直接拟合法，还是扩展法 ****************** #
            # TODO： 这里需要添加判断一开始是左边还是右边
            if len(max_contour_edge) != 0:
                max_contour_edge2 = max_contour_edge[max_contour_edge[:, 0] > max(edge_index[:, 0])]  # edge2
                # extend_h=100
                if len(max_contour_edge2):
                    area = sum(max_contour_edge2[:, 1])
                    # print("{}的顶端面积是{}".format(i, area))
                    if flag2 == 1:
                        if area > T_area:
                            edge_index = np.argwhere(edges == 255)
                            edge_index[:, [0, 1]] = edge_index[:, [1, 0]]

                            if len(edge_index) != 0:
                                y1_2 = edge_index[:, 0] * a2 + b2 - extend_h
                                edge_index = edge_index[y1_2 < edge_index[:, 1]]
                                y2_2 = edge_index[:, 0] * a2 + b2 + extend_h
                                edge_index = edge_index[edge_index[:, 1] < y2_2]
                                if len(edge_index) != 0:
                                    # 画出前个检测直线的扩展区间
                                    # cv2.line(img, (0, int(b2) - extend_h), (w, int(w * a2 + b2 - extend_h)),
                                    #          (0, 255, 0), 3)
                                    # cv2.line(img, (0, int(b2) + extend_h), (w, int(w * a2 + b2 + extend_h)),
                                    #          (0, 255, 0), 3)

                                    a2, b2 = fit_line_by_ransac(edge_index, sigma=3)  # TODO: sigma
                                    cv2.line(img_resize, (0, int(b2)), (w, int(w * a2 + b2)), (255, 0, 0), 3)
                            else:
                                flag2 = 0
                        else:
                            flag2 = 0
                    else:
                        if area > T_area:
                            a2, b2 = fit_line_by_ransac(max_contour_edge2, sigma=3)  # TODO: sigma
                            cv2.line(img_resize, (0, int(b2)), (w, int(w * a2 + b2)), (255, 0, 0), 3)
                            flag2 = 1

            # ********** 作图 ********** #
            name_dir = 'result/Boundary'
            if not os.path.exists(name_dir):
                os.mkdir(name_dir)
            cv2.imwrite(name_dir + '/{}.jpg'.format(i), img_resize)

            i = i + 1

        else:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    end = time.clock()
    print("final is in ", end - start)
    cap.release()
