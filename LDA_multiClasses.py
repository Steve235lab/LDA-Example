# -*- coding: utf-8 -*-
"""
LDA_multiClasses.py

Created on Sat Dec 25 14:10:56 2021

@author: Steve D. J.

Copyright (c) 2021 Steve D. J.. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # 定义数据集：把西瓜数据集随便分成了三类
    X = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], 
                  [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042], [0.719, 0.103]])
    Y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0])
    
    # 计算各维度特征值的均值
    X_mean = [0, 0]
    X_mean[0] = X[:, 0].mean()
    X_mean[1] = X[:, 1].mean()
    μ = np.array(X_mean)
    
    # 计算全局散度矩阵
    St = np.dot((X - μ).T, (X - μ))
        
    print("St:\n", St)
    
    # 计算各类别的类内散度矩阵
    X1 = X[0:5]
    X2 = X[5:11]
    X3 = X[11:]
    x1_mean = [0, 0]
    x2_mean = [0, 0]
    x3_mean = [0, 0]
    x1_mean[0] = X1[:, 0].mean()
    x1_mean[1] = X1[:, 1].mean()
    x2_mean[0] = X2[:, 0].mean()
    x2_mean[1] = X2[:, 1].mean()
    x3_mean[0] = X3[:, 0].mean()
    x3_mean[1] = X3[:, 1].mean()
    μ1 = np.array(x1_mean)
    μ2 = np.array(x2_mean)
    μ3 = np.array(x3_mean)
    
    Sw1 = np.dot((X1 - μ1).T, (X1 - μ1))  
    Sw2 = np.dot((X2 - μ2).T, (X2 - μ2))  
    Sw3 = np.dot((X3 - μ3).T, (X3 - μ3))  
        
    print("Sw1:\n", Sw1)
    print("Sw2:\n", Sw2)            
    print("Sw3:\n", Sw3)

    # 对各类别的类内散度矩阵求和得类内散度矩阵Sw
    Sw = Sw1 + Sw2 + Sw3
    
    print("Sw:\n", Sw)
    
    # 计算类间散度矩阵
    Sb = St - Sw
    
    print("Sb:\n", Sb)
    
    # 计算 Sw^-1S_b 的特征值与特征矩阵
    Sw_inv = np.linalg.inv(Sw)
    eig_vals, eig_vecs = np.linalg.eig(np.dot(Sw_inv, Sb))  # 计算特征值与特征矩阵 详见https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    sorted_indices = np.argsort(eig_vals)                   # 按eig_vals进行排序：sorted_indices = [0, 1]
    w = eig_vecs[:, sorted_indices[:-3:-1]]                 # 取排序后的较大的N-1列作为w，这里三分类，所以取两列
    print("eig_vecs: ", eig_vecs)
    print("w: ", w)                                         # 观察发现，这样得到的w是0、1列调换后的eig_vecs
    #eig_vals, w = np.linalg.eig(np.dot(Sw_inv, Sb))        # 直接让w = eig_vecs得到映射结果
    
    # 计算映射后的值
    z = []
    for i in range(0, Y.size):
        z.append(np.dot(w.T, X[i]))
        
    print("z: ", z)
    
    # 画图
    for i in range(0, 5):
        plt.scatter(z[i][0], z[i][1], color = 'r')
        
    for i in range(5, 11):
        plt.scatter(z[i][0], z[i][1], color = 'g')
    
    for i in range(11, len(Y)):
        plt.scatter(z[i][0], z[i][1], color = 'b')
        
    plt.savefig("LDA_multiClasses_w列调换.jpg",dpi=400,bbox_inches='tight')


    
    

    
