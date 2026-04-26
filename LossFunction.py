# -*- coding:utf-8 -*-
"""
损失函数定义
包含交叉熵损失和PolyLoss（多项式损失）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    """
    PolyLoss：多项式损失函数
    在交叉熵损失的基础上添加多项式项，用于处理类别不平衡问题
    """
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        """
        初始化PolyLoss
        
        Args:
            weight_loss: 类别权重（用于处理类别不平衡）
            DEVICE: 计算设备
            epsilon: 多项式项的权重系数
        """
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        """
        计算PolyLoss
        
        Args:
            predicted: 模型预测值 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            
        Returns:
            loss: 平均损失值
        """
        # 创建one-hot编码
        one_hot = torch.zeros((labels.shape[0], 2), device=self.DEVICE).scatter_(
            1, torch.unsqueeze(labels, dim=-1), 1)
        # 计算预测概率（对真实类别的预测概率）
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        # 交叉熵损失
        ce = self.CELoss(predicted, labels)
        # PolyLoss = CE + epsilon * (1 - pt)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class CELoss(nn.Module):
    """
    标准交叉熵损失函数
    支持类别权重以处理类别不平衡问题
    """
    def __init__(self, weight_CE, DEVICE):
        """
        初始化交叉熵损失
        
        Args:
            weight_CE: 类别权重
            DEVICE: 计算设备
        """
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        """
        计算交叉熵损失
        
        Args:
            predicted: 模型预测值 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            
        Returns:
            loss: 损失值
        """
        return self.CELoss(predicted, labels)
