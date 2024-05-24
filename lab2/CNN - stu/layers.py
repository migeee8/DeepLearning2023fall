import numpy as np


def affine_forward(x, w, b):
    """
    计算全连接神经网络的前向传播
    Args:
        x: Input data (N, D), where N is the number of samples and D is the input feature dimension.
        w: Weight parameters (D, M), where D is the input feature dimension, and M is the output feature dimension.
        b: Bias parameters (M,)

    Returns:
    - out: Output data (N, M), where N is the number of samples, and M is the output feature dimension.
    - cache: A tuple (x, w, b) for caching input data x, weight parameters w, and bias parameters b for use in backward pass.
    """
    num = x.shape[0]
    x_row = x.reshape(num, -1)  # 自动匹配到对应形状
    out = np.dot(x_row, w) + b  # wx + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    计算全连接神经网络的反向传播

    Args:
        dout: Gradient of the loss with respect to the output of this layer.
        cache: A tuple (x, w, b) containing the input data x, weight parameters w, and bias parameters b.

    Returns:
    - dx: Gradient of the loss with respect to the input data x.
    - dw: Gradient of the loss with respect to the weight parameters w.
    - db: Gradient of the loss with respect to the bias parameters b.
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)  # 使dx 与 x 的形状相匹配
    x_row = x.reshape(x.shape[0], -1)  # x转换为行，方便进行点乘
    dw = np.dot(x_row.T, dout)
    db = np.sum(dout, axis=0, keepdims=True)
    return dx, dw, db


def relu_forward(x):
    """
    输入任意x求对应relu值，返回输出和输入的x
    """
    out = ReLU(x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    relu函数的反向传播
    """
    dx, x = None, cache
    dx = dout
    dx[x <= 0] = 0  # relu函数中当x小于等于零的时候导数为0，大于零的时候导数为1

    return dx


def ReLU(x):
    """ReLU non-linearity"""
    return np.maximum(0, x)


def conv_forward_naive(x, w, b, conv_param):
    """
    卷积层前向传播
    Args:
        w: Weights
        b: biases
        x:Input to the convolutional layer
        conv_param:parameters for the convolutional layer

    Returns:
        out, cache:output matrix and history cache(input parameters)
    """
    stride, pad = conv_param['stride'], int(conv_param['pad'])
    num, channel, height, width = x.shape
    filter_num, channel, filter_height, filter_width = w.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    h_new = int(1 + (height + 2 * pad - filter_height) / stride)
    w_new = int(1 + (width + 2 * pad - filter_width) / stride)  # 计算特征图像的size
    s = stride
    out = np.zeros((num, filter_num, h_new, w_new))  # 图像数量*filter数量（也是得到的特征图像数量）*特征图像size（h_new * w_new）

    for i in range(num):  # ith image
        for f in range(filter_num):  # fth filter
            for j in range(h_new):
                for k in range(w_new):
                    out[i, f, j, k] = np.sum(
                        x_padded[i, :, j * s:filter_height + j * s, k * s:filter_width + k * s] * w[f]) + b[f]
                    # 从输入数据 x_padded 中取出与卷积核对应的区域，然后将其与卷积核 w[f] 逐元素相乘，并求和，最后加上偏置 b[f]。
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    卷积层的反向传播

    Args:
        dout: gradient propagated from last layer
        cache: tuple with input x, w, b

    Returns:
    - dx: gradient of x
    - dw: gradient of w
    - db: gradient of b

    """
    x, w, b, conv_param = cache
    pad = int(conv_param['pad'])
    stride = conv_param['stride']
    num, channel, height, width = x.shape
    filter_num, channel, filter_height, filter_width = w.shape
    h_new = int(1 + (height + 2 * pad - filter_height) / stride)
    w_new = int(1 + (width + 2 * pad - filter_width) / stride)

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    for i in range(num):  # ith image
        for f in range(filter_num):  # fth filter
            for j in range(h_new):
                for k in range(w_new):
                    window = x_padded[i, :, j*s:filter_height + j*s, k*s:filter_width + k*s]
                    db[f] += dout[i, f, j, k]
                    dw[f] += window * dout[i, f, j, k]
                    dx_padded[i, :, j*s:filter_height + j*s, k*s:filter_width + k*s] += w[f] * dout[i, f, j, k]
                    # 计算输入数据 x 的梯度 ,用dout中的累积梯度乘以输入的w

    dx = dx_padded[:, :, pad:pad + height, pad:pad + width]  # 最后返回的结果去掉padding

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    池化层前向传播
    Args:
        x:Input to the pooling layer
        pool_param:parameters for the pooling layer

    Returns:
        out, cache:output matrix and history cache(input parameters)
    """
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    s = stride
    num, channel, height, width = x.shape
    h_new = int(1 + (height - pool_height) / s)
    w_new = int(1 + (width - pool_width) / s)  # 计算池化后的尺寸
    out = np.zeros((num, channel, h_new, w_new))
    for i in range(int(num)):  # ith image
        for j in range(int(channel)):  # jth channel
            for k in range(int(h_new)):
                for l in range(int(w_new)):
                    window = x[i, j, k * s:pool_height + k * s, l * s:pool_width + l * s]
                    out[i, j, k, l] = np.max(window)  # 窗口中的最大值
    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    """

    计算最大池化层的反向传播

    """
    x, pool_param = cache;
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    s = stride
    num, channel, height, width = x.shape
    h_new = int(1 + (height - pool_height) / s)
    w_new = int(1 + (width - pool_width) / s)
    dx = np.zeros_like(x)
    for i in range(int(num)):
        for j in range(int(channel)):
            for k in range(int(h_new)):
                for l in range(int(w_new)):
                    window = x[i, j, k * s:pool_height + k * s, l * s:pool_width + l * s]
                    m = np.max(window)
                    dx[i, j, k * s:pool_height + k * s, l * s:pool_width + l * s] = (window == m) * dout[i, j, k, l]
                    #  将窗口中最大值的地方赋值为最大值*累计梯度，其余部分为0
    return dx


def softmax_loss(x, y):
    """
    返回softmax分类器的loss值和loss的梯度值

    Args:
        x: Input scores (N, C), where N is the number of samples, and C is the number of classes.
        y: Ground truth labels (N,) for each sample.

    Returns:
    - loss: Scalar value representing the softmax loss.
    - dx: Gradient of the loss with respect to the input scores x.

    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    num = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(num), y])) / num  # 平均损失
    dx = probs.copy()
    dx[np.arange(num), y] -= 1
    dx /= num  # 平均梯度
    return loss, dx
