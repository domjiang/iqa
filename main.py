import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import tensorflow as tf
import math
import tensorflow.keras.backend as K

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

(x_train, _), (x_test, _) = fashion_mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)
print(x_test.shape)

latent_dim = 1568


# 定义网络
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1568, activation='relu',
                         kernel_regularizer=regularizers.l2(3e-3)),
            layers.Dense(latent_dim, activation='relu',
                         kernel_regularizer=regularizers.l2(3e-3))  # 正则化项就已经加进去了，会体现在最后的loss function上
            # 输出的大小为lantent_dim（隐藏层的大小），激活函数为relu
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(1568, activation='sigmoid',
                         kernel_regularizer=regularizers.l2(3e-3)),
            layers.Dense(784, activation='sigmoid',
                         kernel_regularizer=regularizers.l2(3e-3)),
            layers.Reshape((28, 28))
        ])

    # 上面的代码在实例化的时候就已经被运行了
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 实例化
autoencoder = Autoencoder(latent_dim)


# 定义loss function
def logfunc(x1, x2):
    return tf.math.multiply(x1, tf.math.log(tf.truediv(x1, x2)))


def kl_div(rho, rho_hat):
    term2_num = tf.constant(1.) - rho
    term2_den = tf.constant(1.) - rho_hat
    kl = logfunc(rho, rho_hat) + logfunc(term2_num, term2_den)
    return kl


def design_lossfunc(y_true, y_pred):
    rm = K.sum(tf.pow(y_true - y_pred, 2))
    beta = 5
    kl_loss = K.sum(kl_div(K.mean(autoencoder.encoder(x_test)), 0.035))
    return rm + beta * kl_loss


'''
print(type(autoencoder.encoder(x_test)))
print(len(autoencoder.encoder.layers[1].get_weights()))  # 返回值是2，说明这个list有两个元素，是 kernel 和 bias
print(type(autoencoder.encoder.layers[1].get_weights()[0]))  # 是np.array
print(type(autoencoder.encoder.layers[1].get_weights()[1]))  # 是np.array
print(autoencoder.encoder.layers[1].get_weights()[0].shape)  # array 的size是784，1568 这个应该改是kernel
print(autoencoder.encoder.layers[1].get_weights()[1].shape)  # array 的size是1568，1 这个应该是bias
print(type(autoencoder.encoder.layers[1]))  # 获得第一个sequential层的dense层
print(type(autoencoder.encoder.layers[1].get_weights()))  # 返回的是一个list
print(type(autoencoder.encoder.layers[0]))  # 获得第一个sequential层的Flatten层
print(len(autoencoder.encoder.layers[0].get_weights()))  # array 的size是
'''

# 定义优化器
hh = optimizers.Adam(lr=0.0005)

# 训练
autoencoder.compile(optimizer=hh, loss=design_lossfunc)  # model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
history = autoencoder.fit(x_train, x_train,
                epochs=10,  # 整个的轮次
                shuffle=True,
                validation_data=(x_test, x_test))  # 训练模型
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()  # 层的实例是可调用的，它以张量为参数，并且返回一个张量
print(autoencoder.losses)

# 绘图 n是画几张图
n = 2
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    # print(x_test[i].shape)
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

wt = autoencoder.encoder.layers[1].get_weights()[0]
print(np.sum(wt ** 2) * 3e-3)

print('dddd')
plot_model(autoencoder)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


'''
for layer in autoencoder.layers:
    print(autoencoder.layers)
    print(type(layer.get_weights()[0]))
    print(layer.get_weights()[0].shape)
    print(type(layer.get_weights()[1]))
    print(layer.get_weights()[1].shape)
'''

