# -*- coding: utf-8 -*-
"""
Task6_LDA.py

Created on Wed Dec 22 17:17:01 2021

@author: Steve D. J.

Copyright (c) 2021 Steve D. J.. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # 定义数据集：西瓜数据集
    X = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], 
                  [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042], [0.719, 0.103]])
    Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # 计算各维度特征值的均值
    X1 = X[0:8]
    X2 = X[8:]
    x1_mean = [0, 0]
    x2_mean = [0, 0]
    x1_mean[0] = X1[:, 0].mean()
    x1_mean[1] = X1[:, 1].mean()
    x2_mean[0] = X2[:, 0].mean()
    x2_mean[1] = X2[:, 1].mean()
    μ0 = np.array(x1_mean)
    μ1 = np.array(x2_mean)
    
    # 计算类内散度矩阵 Sw
    Sw = np.dot((X1 - μ0).T, (X1 - μ0))
    Sw = Sw + np.dot((X2 - μ1).T, (X2 - μ1))
    
    # 计算w
    Sw_inv = np.linalg.inv(Sw)
    w = np.dot(Sw_inv, (μ0 - μ1))
    
    print("w: ", w)
    
    # 计算映射后的值
    z = []
    for i in range(0, Y.size):
        z.append(np.dot(w, X[i]))
        
    print("映射后的值：", z)
    print("好瓜映射后的平均值：", np.mean(z[0:8]))
    print("坏瓜映射后的平均值：", np.mean(z[8:]))
    
    # 绘制图像
    # 绘制原始数据
    plt.scatter(X[0:8, 0], X[0:8, 1])
    plt.scatter(X[8:, 0], X[8:, 1], marker='x')
    
    # 设定斜率为 b/a，绘制映射后的点与线
    a = 1.5 * w[0]
    b = 1.8 * w[1]
    z_x = []
    for i in z:
        z_x.append(i * a) 
    z_y = []
    for i in z:
        z_y.append(i * b)
    plt.scatter(z_x[0:8], z_y[0:8], marker='^', color='r')
    plt.scatter(z_x[8:], z_y[8:], marker='^', color='k')
    plt.plot(z_x, z_y)
    for i in range(0, Y.size):
        plt.plot([X[i, 0], z_x[i]], [X[i, 1], z_y[i]], color='r', linestyle = '--')
    
    # 保存图片
    plt.title('LDA')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("LDA.jpg",dpi=400,bbox_inches='tight') 
       
    plt.show()
    
    # 看起来太乱了，再绘制一张简明的图
    for i in z[0:8]:
        plt.scatter(i, 0.02, color='r')
    for i in z[8:]:
        plt.scatter(i, 0, color='k')
        
    plt.ylim([-0.02, 0.04])
    plt.savefig("LDA_zOnly.jpg",dpi=400,bbox_inches='tight')
    
    plt.show()
    


    
    
                    