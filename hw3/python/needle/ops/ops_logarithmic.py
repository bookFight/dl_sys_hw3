from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    """对输入应用对数Softmax操作，数值稳定版本"""

    def compute(self, Z):
        """
        前向计算：LogSoftmax(z) = z - LogSumExp(z)
        其中LogSumExp在最后一个轴上计算（默认对特征维度操作）
        """
        # 计算输入在最后一个轴上的LogSumExp，保持维度以支持广播
        log_sum_exp = logsumexp(Z, axes=-1, keepdims=True)
        # 应用LogSoftmax公式
        return Z - log_sum_exp

    def gradient(self, out_grad, node):
        """
        反向传播：计算LogSoftmax的梯度
        推导：若y = LogSoftmax(z)，则∂L/∂z = ∂L/∂y - sum(∂L/∂y * softmax(z))
        其中softmax(z) = exp(y)
        """
        # 获取输入张量Z
        Z = node.inputs[0].cached_data
        # 计算softmax值：exp(LogSoftmax(Z)) = exp(Z - LogSumExp(Z))
        softmax_val = ndl.exp(node.outputs[0])  # shape与Z一致

        # 确定操作的轴（最后一个轴）
        axes = -1
        # 计算上游梯度与softmax的乘积，并在指定轴上求和（保持维度用于广播）
        sum_term = ndl.summation(out_grad * softmax_val, axes=axes, keepdims=True)
        # 梯度 = 上游梯度 - 求和项（广播后）
        return out_grad - sum_term


def logsoftmax(a):
    """便捷函数：对输入应用LogSoftmax操作"""
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    """
    对数求和指数操作：计算log(sum(exp(z)))，数值稳定版本
    通过减去最大值避免指数爆炸：log(sum(exp(z - max_z))) + max_z
    """
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes  # 要求和的轴
        self.keepdims = keepdims  # 是否保持求和后的维度

    def compute(self, Z):
        """前向计算：数值稳定的LogSumExp"""
        # 计算指定轴上的最大值（保持维度用于广播）
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        # 计算exp(z - max_z)避免数值溢出
        exp_shifted = array_api.exp(Z - max_z)
        # 求和并取对数
        sum_exp = array_api.sum(exp_shifted, axis=self.axes, keepdims=self.keepdims)
        log_sum = array_api.log(sum_exp)
        # 加上之前减去的最大值（需要调整维度以匹配）
        if self.keepdims:
            # 若保持维度，直接使用带维度的max_z求和后广播
            max_z_sum = array_api.sum(max_z, axis=self.axes, keepdims=self.keepdims)
            return log_sum + max_z_sum
        else:
            # 若不保持维度，压缩max_z的维度后相加
            max_z_squeezed = array_api.squeeze(max_z, axis=self.axes)
            return log_sum + max_z_squeezed

    def gradient(self, out_grad, node):
        """
        反向传播：计算LogSumExp的梯度
        推导：∂L/∂z = (exp(z - max_z) / sum(exp(z - max_z))) * ∂L/∂y
        其中y = LogSumExp(z)
        """
        Z = node.inputs[0].cached_data  # 输入数据
        out_grad_data = out_grad.cached_data  # 上游梯度数据

        # 计算输入在指定轴上的最大值（保持维度用于广播）
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        # 计算exp(z - max_z)
        exp_shifted = array_api.exp(Z - max_z)
        # 计算sum(exp(z - max_z))用于归一化
        sum_exp = array_api.sum(exp_shifted, axis=self.axes, keepdims=True)
        # 梯度权重：exp_shifted / sum_exp（即softmax的变体）
        weight = exp_shifted / sum_exp

        # 调整上游梯度的维度以匹配weight（支持广播）
        if self.axes is not None:
            # 确定需要广播的形状（将求和轴设置为1）
            broadcast_shape = list(Z.shape)
            axes = (self.axes,) if isinstance(self.axes, int) else self.axes
            for ax in axes:
                broadcast_shape[ax] = 1
            # 重塑上游梯度以支持广播
            out_grad_reshaped = array_api.reshape(out_grad_data, tuple(broadcast_shape))
        else:
            # 若对所有轴求和，上游梯度是标量，重塑为全1维度
            out_grad_reshaped = array_api.reshape(out_grad_data, (1,) * len(Z.shape))

        # 最终梯度 = 权重 * 重塑后的上游梯度
        grad = weight * out_grad_reshaped
        return Tensor(grad)


def logsumexp(a, axes=None, keepdims=False):
    """便捷函数：对输入应用LogSumExp操作"""
    return LogSumExp(axes=axes, keepdims=keepdims)(a)
