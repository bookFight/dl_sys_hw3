import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p  # 翻转概率，默认50%

    def __call__(self, img):
        # 输入img为 H x W x C 形状的NDArray（高度×宽度×通道数）
        # 生成随机数，若小于p则执行翻转
        flip_img = np.random.rand() < self.p
        if flip_img:
            # 水平翻转：对宽度维度（W）进行逆序切片（::-1）
            # 例如 [:, ::-1, :] 表示保留高度（H）和通道（C）不变，翻转宽度
            img = img[:, ::-1, :]
        return img  # 返回翻转后或原图

class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding  # 四周填充的像素数，默认3

    def __call__(self, img):
        # 输入img为 H x W x C 形状的NDArray
        # 生成随机偏移量：x和y方向的偏移范围为[-padding, padding]
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding + 1, size=2)

        H, W, C = img.shape  # 获取原图高度、宽度、通道数
        # 1. 创建带填充的图像：四周各加padding，形状为 (H+2p) x (W+2p) x C
        pad = np.zeros((H + 2 * self.padding, W + 2 * self.padding, C))
        # 将原图放入填充图像的中心区域
        pad[self.padding:self.padding + H, self.padding:self.padding + W, :] = img