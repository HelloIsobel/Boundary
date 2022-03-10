import pandas as pd
import random
import math
import cv2
import numpy as np
import argparse



def getExcel(features, excel_path, header=True):
    """保存values到excel文件中"""
    data = pd.DataFrame(features)
    writer = pd.ExcelWriter(excel_path)
    if header:
        data.to_excel(writer, 'page_1', float_format='%.5f')
    else:
        data.to_excel(writer, 'page_1', float_format='%.5f', header=False, index=False)
    writer.save()
    writer.close()


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


def get_maxContour(bw, T_contour_area):
    """得到最大面积连通域"""
    pred_flip = 255 - bw  # 因为下面检测contours默认是包围白色像素点，也就是把黑色像素点当做背景
    pred_flip = pred_flip.astype("uint8")
    contours, hierarchy = cv2.findContours(pred_flip, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # global T_contour_area
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