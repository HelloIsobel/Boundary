import cv2
import os
import numpy as np
import glob
import random
import math


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

    global w
    global h
    contours_area = []
    max_edge_no_boundary = np.array([])

    if len(contours) != 0:
        for j in range(len(contours)):
            contours_area.append(cv2.contourArea(contours[j]))
        max_area_index = np.argmax(np.array(contours_area))

        if contours_area[int(max_area_index)] > 30000:  # 原来这里是50000 但是将比较小的有效值扔掉了 所以改为20000
            max_edge = np.squeeze(contours[max_area_index])
            index = np.argwhere(max_edge[:, 1] == 0)
            index = np.append(index, np.argwhere(max_edge[:, 0] == 0))
            index = np.append(index, np.argwhere(max_edge[:, 0] == w - 1))
            index = np.append(index, np.argwhere(max_edge[:, 1] == h - 1))
            # index = np.sort(index)
            index = np.append(np.sort(index), [len(max_edge) - 1])
            index_sub = np.array([0])
            index_sub = np.append(index_sub, index[0:-1])  # 这里是倒数第二个
            sub = index - index_sub
            max_index = np.argwhere(sub == max(sub))
            max_edge_no_boundary = max_edge[int(index[max_index - 1]):int(index[max_index]), :]

    return max_edge_no_boundary


if __name__ == "__main__":
    # path = 'D:/Desktop/test/select_IoU/' + "2"
    # dirpath = glob.glob('D:/Desktop/Paper_ Invention/newResult_Boundary_detection/Data/all_train_test/*.jpg')
    dirpath = glob.glob(r'/home/zxy/3-Boundary_detection/Unet/data/all_train_test/*.jpg')
    # 遍历所有图片
    flag = 0  # 判断是否使用扩展检测边界线法
    a, b = 0, 0  # 检测直线的斜率和截距

    flag2 = 0
    a2, b2 = 0, 0
    T_area = 60000

    for img_path in dirpath:
        # 保存结果地址
        name = img_path.split('\\')[-1]

        # 读取图片
        img_src = cv2.imread(img_path)
        w = img_src.shape[1]
        h = img_src.shape[0]

        pred_path = img_path.replace("src", "pred")
        pred = cv2.imread(pred_path, 0)

        # # # ********
        # # # 画图，预测红色透明面积
        # # # ********
        # index = np.argwhere(pred == 0)
        # for i in range(len(index)):
        #     x = index[i, 0]
        #     y = index[i, 1]
        #
        #     # # ******** red
        #     if img[x, y, 2] > 205:
        #         img[x, y, 2] = 255
        #     else:
        #         img[x, y, 2] = img[x, y, 2] + 50
        #
        #     img[x, y, 0] = img[x, y, 0] / 1.5
        #     img[x, y, 1] = img[x, y, 1] / 1.5

        kernel = np.ones((2, 2), np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)  # 先腐蚀再膨胀 这里对应闭运算
        # ************ 最大轮廓 ************ #
        max_contour_edge = get_maxContour(pred)

        # # # ********
        # # # ******** 画图，预测红色透明面积+候选边界线
        # # # ********
        # for coor in max_contour_edge:
        #     cv2.circle(img, (int(coor[0]), int(coor[1])), 1, (0, 255, 0), 4)  # ( 0, 255, 255)  (255, 0, 0)
        #
        # name_dir = path+'/candidateBoundary'
        # if not os.path.exists(name_dir):
        #     os.mkdir(name_dir)
        # cv2.imwrite(name_dir + '/{}'.format(name), img)

        # # ********
        # ******** 画图，最终边界线
        # # ********

        # ******************************** 检测边界线 ******************************* #
        edge_index = np.array([[w + 10, 0]])
        # opening = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)
        # 提取图像边缘
        merge_G = cv2.GaussianBlur(pred, (3, 3), 0)
        merge_G = merge_G.astype("uint8")
        edges = cv2.Canny(merge_G, 10, 50, apertureSize=3)

        # ****************** 判断直线1 采用直接拟合法，还是扩展法 ****************** #
        extend_w = 100  # TODO: 扩展宽度
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
                    # cv2.line(frame, (0, int(b) - extend_h), (w, int(w * a + b - extend_h)), (0, 255, 0), 3)
                    # cv2.line(frame, (0, int(b) + extend_h), (w, int(w * a + b + extend_h)), (0, 255, 0), 3)

                    a, b = fit_line_by_ransac(edge_index, sigma=3)  # TODO: sigma
                    cv2.line(img_src, (0, int(b)), (w, int(w * a + b)), (0, 0, 255), 5)
                else:
                    flag = 0
            else:
                flag = 0
        else:
            # ****************** 直接法 ****************** #
            if len(max_contour_edge) != 0:
                a, b = fit_line_by_ransac(max_contour_edge, sigma=3)  # TODO: sigma
                cv2.line(img_src, (0, int(b)), (w, int(w * a + b)), (0, 0, 255), 5)
                flag = 1

        # ****************** 判断直线2 采用直接拟合法，还是扩展法 ****************** #
        # TODO： 这里需要添加判断一开始是左边还是右边
        if len(max_contour_edge) != 0:
            max_contour_edge2 = max_contour_edge[max_contour_edge[:, 0] > max(edge_index[:, 0])]  # edge2
            # extend_h=100
            if len(max_contour_edge2):
                area = sum(max_contour_edge2[:, 1])
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
                                # cv2.line(frame, (0, int(b2) - extend_h), (w, int(w * a2 + b2 - extend_h)),
                                #          (0, 255, 0), 3)
                                # cv2.line(frame, (0, int(b2) + extend_h), (w, int(w * a2 + b2 + extend_h)),
                                #          (0, 255, 0), 3)

                                a2, b2 = fit_line_by_ransac(edge_index, sigma=3)  # TODO: sigma
                                cv2.line(img_src, (0, int(b2)), (w, int(w * a2 + b2)), (0, 0, 255), 5)
                        else:
                            flag2 = 0
                    else:
                        flag2 = 0
                else:
                    if area > T_area:
                        a2, b2 = fit_line_by_ransac(max_contour_edge2, sigma=3)  # TODO: sigma
                        cv2.line(img_src, (0, int(b2)), (w, int(w * a2 + b2)), (0, 0, 255), 5)
                        flag2 = 1

        name_dir = path + '/line'
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        cv2.imwrite(name_dir + '/{}'.format(name), img_src)
