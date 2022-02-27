import glob
import numpy as np
import torch
import os
import cv2
import time
from newUnet_our.new_unet_model import U_net
import pandas as pd


def p_r_F1_PA_mIoU(pred, label):
    TP, TN, FN, FP = 0, 0, 0, 0
    TP = ((pred == 255) & (label == 255)).sum()
    TN = ((pred == 0) & (label == 0)).sum()
    FN = ((pred == 0) & (label == 255)).sum()
    FP = ((pred == 255) & (label == 0)).sum()

    p = TP / (TP + FP)  # 精确率 precision
    r = TP / (TP + FN)  # 召回率 recall

    p1 = TN / (TN + FN)  # 精确率 precision
    r1 = TN / (TN + FP)  # 召回率 recall

    F1_ = 2 * r * p / (r + p)  # F1-score
    PA = (TP + TN) / (TP + TN + FP + FN)  # 准确率 pixel accuracy(PA)
    # mIoU
    IoU1 = TP / (FP + FN + TP)
    IoU2 = TN / (TN + FN + FP)
    mIoU = (IoU1 + IoU2) / 2

    return TP, TN, FN, FP, IoU1, IoU2, mIoU, p, r, p1, r1


if __name__ == "__main__":
    start1 = time.clock()  # 总时间
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = U_net(1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(
        torch.load('/home/zxy/3-Boundary_detection/new_best_model_5kinds_272_epoch100.pth', map_location=device))

    # 测试模式
    net.eval()
    # 读取所有图片路径
    mAP_all_excel = '/home/zxy/3-Boundary_detection/Unet/mAP/newUnet_mAP_all.xlsx'
    path = glob.glob('/home/zxy/3-Boundary_detection/Unet/data/all_train_test/*.jpg')

    # all_train_acc = []
    all_mAP_4 = []
    zhibiao_iou_all=[]
    for i in range(10):
        a = []
        all_train_acc = []
        zhibiao_all=[]
        zhibiao=[]
        num = int(i + 1) * 0.1
        mAP_excelpath = '/home/zxy/3-Boundary_detection/Unet/mAP/newUnet_mAP_' + str(num) + '.xlsx'
        print("This is --" + str(num))
        for img_path in path:
            # 保存结果地址
            # save_res_path = img_path.replace("all_train_test", "pred_1011newUnet_dilation1")
            label_path = img_path.replace("all_train_test", "all_train_test_label")
            name1 = label_path.split('/')[-1]
            name = name1.split('.')[0]

            # 读取图片
            img_src = cv2.imread(img_path)
            # 转为灰度图
            img = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)  # TODO:改成3通道需要注释掉
            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])  # TODO:改成3通道修改修改
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # 预测
            # start = time.clock()
            pred = net(img_tensor)

            # 提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            # 处理结果
            pred[pred >= num] = 255
            pred[pred < num] = 0

            # p_r_F1_PA_mIoU(pred, label)
            img_label = cv2.imread(label_path, 0)
            train_acc = p_r_F1_PA_mIoU(pred, img_label)  # TODO:召回率

            a = [name, train_acc[0], train_acc[1], train_acc[2], train_acc[3], train_acc[4], train_acc[5], train_acc[6],
                 train_acc[7], train_acc[8], train_acc[9], train_acc[10]]
            all_train_acc.append(a)
            print(name)
            zhibiao = [train_acc[7], train_acc[8], train_acc[9], train_acc[10]]
            zhibiao_all.append(zhibiao)

            data = pd.DataFrame(all_train_acc)
            writer = pd.ExcelWriter(mAP_excelpath)
            data.to_excel(writer, 'page_1', header=None, index=False)
            # header=['name', 'TP', 'TN', 'FN', 'FP', 'IoU1', 'IoU2', 'mIoU', 'p', 'r', 'p1', 'r1'],

            data.to_excel(writer, 'page_1', header=False, index=False)
            writer.save()
            writer.close()

        # # 算平均
        zhibiao_m = np.mean(zhibiao_all, axis=0)
        zhibiao_iou = [num, zhibiao_m[0], zhibiao_m[1], zhibiao_m[2], zhibiao_m[3]]
        zhibiao_iou_all.append(zhibiao_iou)

        data2 = pd.DataFrame(zhibiao_iou_all)
        writer2 = pd.ExcelWriter(mAP_all_excel)
        # data2.to_excel(writer2, 'page_1', header=['iou', 'p', 'r', 'p1', 'r1'], index=False)
        data2.to_excel(writer2, 'page_1', header=None, index=False)
        writer2.save()
        writer2.close()
