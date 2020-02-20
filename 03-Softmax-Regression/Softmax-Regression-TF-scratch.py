import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
batch_size = 256
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
x_train = x_train / 255.0
x_test = x_test / 255.0
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


class SoftmaxRegression:
    def __init__(self, num_inputs, num_outputs):
        self.Weight = tf.Variable(
            tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
        self.Bias = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))

    def __call__(self, x):
        logits = tf.matmul(tf.reshape(x, shape=(-1, self.Weight.shape[0])), self.Weight) + self.Bias
        return softmax(logits)


def softmax(logits, axis=-1):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)


def SoftmaxCrossEntropyLoss(y_hat, y_true):
    return tf.losses.sparse_categorical_crossentropy(y_true, y_hat)


# 描述,对于tensorflow2中，比较的双方必须类型都是int型，所以要将输出和标签都转为int型
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y, dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n


num_inputs, num_outputs, num_hiddens = 784, 10, 256
model = SoftmaxRegression(num_inputs, num_outputs)
loss = SoftmaxCrossEntropyLoss

# 这里使用 1e-3 学习率，是因为原文 0.1 的学习率过大，会使 cross_entropy loss 计算返回 numpy.nan
num_epochs, lr = 5, 1e-3

for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        with tf.GradientTape() as tape:
            y_hat = model(X)
            l = tf.reduce_sum(loss(y_hat, y))

        params = [model.Weight, model.Bias]
        grads = tape.gradient(l, params)
        params[0].assign_sub(grads[0] * lr)
        params[1].assign_sub(grads[1] * lr)

        y = tf.cast(y, dtype=tf.float32)
        train_l_sum += l.numpy()
        train_acc_sum += tf.reduce_sum(
            tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
        n += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, model)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
          % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

import matplotlib.pyplot as plt

X, y = iter(test_iter).next()


def get_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_mnist(images, labels):
    # 这⾥的_表示我们忽略（不使⽤）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))  # 这里注意subplot 和subplots 的区别
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(tf.reshape(img, shape=(28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


true_labels = get_mnist_labels(y.numpy())
pred_labels = get_mnist_labels(tf.argmax(model(X), axis=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_mnist(X[0:9], titles[0:9])
