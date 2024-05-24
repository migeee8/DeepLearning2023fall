from layer_utils import *
import numpy as np


class ThreeLayerCovNet(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 d_type=np.float32):
        self.params = {}
        self.reg = reg  # 正则化惩罚项
        self.d_type = d_type  # 数据类型转换为float32

        # 参数矩阵初始化
        img_channel, img_height, img_width = input_dim  # 初始输入图像的通道数， 高， 宽
        self.params['w1'] = weight_scale * np.random.randn(num_filters, img_channel, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params["w2"] = weight_scale * np.random.randn(int(num_filters * img_height * img_width / 4), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['w3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(d_type)

    def loss(self, x, y=None):
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        filter_size = w1.shape[2]  # 卷积核边长
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}  # 卷积核步长设置为1，边缘填充设置为(filter_size - 1) / 2
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}  # 池化核设置为2*2， 步长设置为2

        # 前向传播
        a1, cache1 = conv_relu_pool_forward(x, w1, b1, conv_param, pool_param)
        a2, cache2 = affine_relu_forward(a1, w2, b2)
        scores, cache3 = affine_forward(a2, w3, b3)

        if y is None:
            return scores

        # 反向传播
        data_loss, d_scores = softmax_loss(scores, y)
        da2, dw3, db3 = affine_backward(d_scores, cache3)
        da1, dw2, db2 = affine_relu_backward(da2, cache2)
        dx, dw1, db1 = conv_relu_pool_backward(da1, cache1)

        # 计算正则化惩罚项
        dw1 += self.reg * w1
        dw2 += self.reg * w2
        dw3 += self.reg * w3
        reg_loss = 0.5 * self.reg * sum(np.sum(w * w) for w in [w1, w2, w3])

        loss = data_loss + reg_loss
        grads = {'w1': dw1, 'b1': db1, 'w2': dw2, 'b2': db2, 'w3': dw3, 'b3': db3}

        return loss, grads
