# -*- coding:utf-8 -*-
"""
数据准备工具函数
用于K折交叉验证的数据划分和数据集打乱
"""

import numpy as np

def get_kfold_data(i, datasets, k=5):
    """
    获取K折交叉验证中第i折的训练集和验证集
    
    Args:
        i: 当前折数索引（从0开始）
        datasets: 数据集列表
        k: 总折数，默认为5
        
    Returns:
        trainset: 训练集
        validset: 验证集
    """
    # 计算每折的大小
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    # 根据折数位置确定验证集和训练集
    if i != k - 1 and i != 0:
        # 中间折：验证集在中间，训练集在两端
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        # 第一折：验证集在开头
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        # 最后一折：验证集在末尾
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    """
    使用指定随机种子打乱数据集
    
    Args:
        dataset: 待打乱的数据集
        seed: 随机种子
        
    Returns:
        dataset: 打乱后的数据集（原地修改）
    """
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
