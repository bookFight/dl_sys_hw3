"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
# 偏置
        self.bias = bias
        weight=init.kaiming_uniform(in_features,out_features,nonlinearity="relu", dtype=dtype)
        self.weight = Parameter(weight)
        if bias:
            self.bias = Parameter(init.kaiming_uniform(fan_in=out_features,  # 注意这里fan_in为输出特征数
                                    fan_out=1,  # 偏置是一维的，fan_out设为1不影响
                                    nonlinearity="relu",
                                    dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 计算矩阵乘法: x (N, in_features) × weight (in_features, out_features) → (N, out_features)
        out=ops.matmul(X,self.weight)
        if self.bias:
            bias=ops.broadcast_to(self.bias,out.shape)
            out=ops.add(out,bias)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # 获取输入张量的第一个维度（通常是批次大小，如batch_size）
        n = X.shape[0]

        # 计算剩余所有维度的乘积，得到展平后的特征维度
        dim = 1
        for i in range(1, len(X.shape)):  # 从索引1开始遍历（跳过第一个维度）
            dim *= X.shape[i]  # 累积相乘所有后续维度的大小

        # 将输入张量重塑为(batch_size, 展平后的特征数)的二维张量
        return ops.reshape(X, (n, dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x=module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        Z=ops.LogSumExp(axes=(1, ))(logits)
        batch_size=logits.shape[0]
        num_classes=logits.shape[1]
        one_hot=init.one_hot(num_classes,y)

        Z_y=ops.summation(logits*one_hot, axes=(1, ))
        loss=Z-Z_y;
        sumLoss=ops.summation(loss)
        return sumLoss/batch_size
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 确保权重和偏置的形状正确（避免重复重塑操作影响计算图）
        if self.weight.shape != (1, self.dim):
            self.weight = self.weight.reshape((1, self.dim))
        if self.bias.shape != (1, self.dim):
            self.bias = self.bias.reshape((1, self.dim))

        if self.training:
            batch_size, feature_size = x.shape
            # 计算当前批次的均值和方差（会产生计算图）
            mean = (x.sum(axes=(0,)) / batch_size).reshape((1, feature_size))
            var = (((x - mean.broadcast_to(x.shape)) ** 2).sum(axes=(0,)) / batch_size).reshape((1, feature_size))

            # 关键：用detach()分离计算图，避免running_mean/var关联批量统计量的计算图
            # 运行时统计量仅记录状态，不参与梯度回传
            self.running_mean = self.running_mean * (1 - self.momentum) + mean.detach().reshape(
                self.running_mean.shape) * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + var.detach().reshape(
                self.running_var.shape) * self.momentum

            # 广播均值和方差到输入形状，用于当前批次的归一化
            mean_broadcast = mean.broadcast_to(x.shape)
            var_broadcast = var.broadcast_to(x.shape)

            # 计算归一化结果及最终输出
            std_x = (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5)
            weight_broadcast = self.weight.broadcast_to(x.shape)
            bias_broadcast = self.bias.broadcast_to(x.shape)
            return std_x * weight_broadcast + bias_broadcast
        else:
            # 推理模式：使用运行时统计量，无需计算图
            running_mean_broadcast = self.running_mean.broadcast_to(x.shape)
            running_var_broadcast = self.running_var.broadcast_to(x.shape)
            std_x = (x - running_mean_broadcast) / ops.power_scalar(running_var_broadcast + self.eps, 0.5)
            weight_broadcast = self.weight.broadcast_to(x.shape)
            bias_broadcast = self.bias.broadcast_to(x.shape)
            return std_x * weight_broadcast + bias_broadcast
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(self.dim))
        self.bias=Parameter(init.zeros(self.dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_size = x.shape
        mean = (x.sum(axes=(1,)) / feature_size).reshape((batch_size, 1)).broadcast_to(x.shape)
        var = (((x - mean) ** 2).sum(axes=(1,)) / feature_size).reshape((batch_size, 1)).broadcast_to(x.shape)
        std_x = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
        weight = self.weight.broadcast_to(x.shape)
        bias = self.bias.broadcast_to(x.shape)
        return std_x * weight + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # 仅在训练模式下应用dropout
        if self.training:
            # 生成掩码：随机生成与x形状相同的0/1张量，其中1的概率为(1-p)
            # 除以(1-p)是为了保持输出的期望值不变（缩放补偿）
            mask = init.randb(*x.shape, p=1 - self.p)
            # 将输入与掩码相乘，实现随机丢弃（被掩码为0的元素被丢弃）
            return x * mask/ (1 - self.p)
        # 推理模式下不做任何处理，直接返回输入
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION
