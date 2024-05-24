import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient


class MultiLayerPerceptron:
    def __init__(self, data, labels, layers, normalize_data=False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]  # 数据预处理
        self.data = data_processed  # 存储处理后的数据矩阵，训练和预测使用的输入数据
        self.labels = labels  # 存储标签数据 初始格式类似[5,7,3,1,0]，存储每幅手写图像对应的数值
        self.layers = layers  # 存储神经网络的层结构，其中包含每一层的神经元数量 [784, 25, 10]
        self.normalize_data = normalize_data  # 是否对数据进行归一化
        self.thetas = MultiLayerPerceptron.thetas_init(layers)  # 初始化权重矩阵，这些权重将在训练过程中不断更新

    def train(self, max_iterations=1000, alpha=0.1):
        unrolled_theta = MultiLayerPerceptron.thetas_unroll(self.thetas)  # 权重矩阵展开成一维向量
        # 传递训练数据
        optimized_theta, cost_history = MultiLayerPerceptron.gradient_descent(self.data, self.labels, unrolled_theta,
                                                                              self.layers, max_iterations, alpha)
        self.thetas = MultiLayerPerceptron.thetas_roll(optimized_theta, self.layers)  # 重新转换成权重矩阵
        return self.thetas, cost_history

    def predict(self, data):
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]
        predictions = MultiLayerPerceptron.feedforward_propagation(data_processed, self.thetas, self.layers)
        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)
        thetas = {}  # 创建字典，存储权重参数的矩阵，在3层神经网络中，该字典中存储2个矩阵
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05  # 随机参数矩阵 25*785 26*10
        return thetas

    @staticmethod
    def thetas_unroll(thetas):  # 矩阵展开为一维向量形式
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([])  # 新建一维向量
        for theta_layer_index in range(num_theta_layers):
            unrolled_theta = np.hstack((unrolled_theta, thetas[theta_layer_index].flatten()))
        return unrolled_theta

    @staticmethod
    def thetas_roll(unrolled_thetas, layers):  # 一维向量还原为矩阵形式
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]

            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume  # 获取当前层和下一层的神经元数量，以及当前层权重矩阵的宽度和高度来计算总偏移量
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height, thetas_width))
            unrolled_shift = unrolled_shift + thetas_volume

        return thetas

    @staticmethod
    def gradient_descent(data, labels, unrolled_theta, layers, max_iterations, alpha):
        optimized_theta = unrolled_theta  # optimised _theta为每次迭代的权重，此处是赋初值操作
        cost_history = []  # 存储每次迭代后的损失值
        for _ in range(max_iterations):
            # 前向传播，求损失
            cost = MultiLayerPerceptron.cost_function(data, labels,
                                                      MultiLayerPerceptron.thetas_roll(optimized_theta, layers), layers)
            cost_history.append(cost)
            # 反向传播，求梯度，用梯度下降迭代参数矩阵
            theta_gradient = MultiLayerPerceptron.gradient_step(data, labels, optimized_theta, layers)
            optimized_theta = optimized_theta - alpha * theta_gradient  # 更新
        return optimized_theta, cost_history

    @staticmethod
    def cost_function(data, labels, thetas, layers):
        num_examples = data.shape[0]  # 训练数据的样本数量
        num_labels = layers[-1]  # 标签数量

        predictions = MultiLayerPerceptron.feedforward_propagation(data, thetas, layers)  # 返回每个样本属于每个类别的概率的矩阵
        bitwise_labels = np.zeros((num_examples, num_labels))  # 将labels转换为格式与上面相统一的01矩阵用来比对和计算误差值
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))  # 正确分类的损失
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))  # 未正确分类的损失
        cost = (-1 / num_examples) * (bit_set_cost + bit_not_set_cost)  # 平均损失，-1是上面的对数损失函数中的
        return cost

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        theta = MultiLayerPerceptron.thetas_roll(optimized_theta, layers)
        thetas_rolled_gradient = MultiLayerPerceptron.back_propagation(data, labels, theta, layers)
        thetas_unrolled_gradients = MultiLayerPerceptron.thetas_unroll(thetas_rolled_gradient)
        return thetas_unrolled_gradients

    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]  # 样本数
        in_layer_activation = data  # 第一层输入值赋初值

        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]  # 获取该层权重
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))  # 激活值 = sigmoid（当前层输入*权重）
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))  # 激活值矩阵 + 偏置项
            in_layer_activation = out_layer_activation  # 下一层输入 = 上一层激活值

        return in_layer_activation[:, 1:]  # 第一列是偏置项，只要结果所以屏蔽掉

    @staticmethod
    def back_propagation(data, labels, thetas, layers):
        num_layers = len(layers)  # 神经网络的总层数
        num_examples, num_features = data.shape  # 样本数 特征数（像素）
        num_label_types = layers[-1]  # label的数量
        deltas = {}  # deltas中存储每一层的梯度

        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            deltas[layer_index] = np.zeros((out_count, in_count + 1))  # deltas初始化
        # 遍历每个样本
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}  # 存储每一层的输入和激活值
            layers_activation = data[example_index, :].reshape((num_features, 1))  # 第一次的激活值直接等于输入值 785*1
            layers_activations[0] = layers_activation

            # 前向传播获得每一层输入和激活值
            for layer_index in range(num_layers - 1):
                layer_theta = thetas[layer_index]
                layer_input = np.dot(layer_theta, layers_activation)  # 点乘
                layers_activation = np.vstack((np.array(([1])), sigmoid(layer_input)))  # 激活函数+偏置参数
                layers_inputs[layer_index + 1] = layer_input
                layers_activations[layer_index + 1] = layers_activation
            output_layer_activation = layers_activation[1:, :]

            delta = {}
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1
            # 最后一层的delta差异 即为输出层的误差delta 激活值与实际标签的差异
            delta[num_layers - 1] = output_layer_activation - bitwise_label

            for layer_index in range(num_layers - 2, 0, -1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index + 1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array(1), layer_input))
                delta[layer_index] = np.dot(layer_theta.T, next_delta) * sigmoid_gradient(layer_input)  # 公式 计算当前层的 delta
                delta[layer_index] = delta[layer_index][1:, :]  # 去掉 delta 中的偏置项
            for layer_index in range(num_layers - 1):
                layer_delta = np.dot(delta[layer_index + 1], layers_activations[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta  # 累加到 deltas 中

        for layer_index in range(num_layers - 1):
            deltas[layer_index] = deltas[layer_index] * (1 / num_examples)  # 平均梯度

        return deltas
