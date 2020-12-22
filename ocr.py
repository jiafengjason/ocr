#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
https://www.cnblogs.com/further-further-further/p/10755361.html
https://blog.csdn.net/weixin_34311757/article/details/92427326
'''

import os
import shutil
import string
import random
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

SAVE_PATH = "model/"
CAPTCHA_IMAGE_PATH = 'images/'

MAX_CAPTCHA = 4
CHAR_SET = list(string.digits + string.ascii_lowercase + string.ascii_uppercase)
CHAR_SET_LEN = len(CHAR_SET)

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

def gen_captcha_text_and_image():
    # 创建图像实例对象
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    # 随机选择4个字符
    captcha_text = ''.join([random.choice(CHAR_SET) for j in range(MAX_CAPTCHA)])
    # 生成验证码
    captcha = image.generate(captcha_text)
    #image.write(captcha_text, CAPTCHA_IMAGE_PATH + captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

# 点降噪
def interference_point(imgName, img, x = 0, y = 0):
    """
    9邻域框,以当前点为中心的田字框,黑点个数
    :param x:
    :param y:
    :return:
    """
    # todo 判断图片的长宽度下限
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    cur_pixel = img[x,y]# 当前像素点的值
    height,width = img.shape[:2]

    for y in range(0, width - 1):
      for x in range(0, height - 1):
        if y == 0:  # 第一行
            if x == 0:  # 左上顶点,4邻域
                # 中心点旁边3个点
                sum = int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右上顶点
                sum = int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            else:  # 最上非顶点,6邻域
                sum = int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 3 * 245:
                  img[x, y] = 0
        elif y == width - 1:  # 最下面一行
            if x == 0:  # 左下顶点
                # 中心点旁边3个点
                sum = int(cur_pixel) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x, y - 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右下顶点
                sum = int(cur_pixel) \
                      + int(img[x, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y - 1])

                if sum <= 2 * 245:
                  img[x, y] = 0
            else:  # 最下非顶点,6邻域
                sum = int(cur_pixel) \
                      + int(img[x - 1, y]) \
                      + int(img[x + 1, y]) \
                      + int(img[x, y - 1]) \
                      + int(img[x - 1, y - 1]) \
                      + int(img[x + 1, y - 1])
                if sum <= 3 * 245:
                  img[x, y] = 0
        else:  # y不在边界
            if x == 0:  # 左边非顶点
                sum = int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])

                if sum <= 3 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右边非顶点
                sum = int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x - 1, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1])

                if sum <= 3 * 245:
                  img[x, y] = 0
            else:  # 具备9领域条件的
                sum = int(img[x - 1, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1]) \
                      + int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 4 * 245:
                  img[x, y] = 0
    cv2.imwrite(CAPTCHA_IMAGE_PATH + imgName + '.jpg',img)
    return img

def otsu(imgName, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imwrite(CAPTCHA_IMAGE_PATH + imgName + '.jpg',th)
    return th

def convert2gray(imgName, img):
    if len(img.shape) > 2:
        # 灰度化, 这里使用的是求均值的方法
        gray = np.mean(img, -1)
        # 上面的转法较快，正规的方法应该是RGB三个通道上按照一定的比例取值
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        #cv2.imwrite(CAPTCHA_IMAGE_PATH + imgName + '.jpg',gray)
        return gray
    else:
        return img

def denoising(image):
    """
    处理图片，方便更好识别，学习
    :param image:图片对象
    :return: 处理之后的图片
    """

    threshold = 128  # 通过设置阈值，去除不必要的干扰物

    for i in range(image.width):
        for j in range(image.height):
            r,g,b = image.getpixel((i,j))
            if (r > threshold or g >threshold or b > threshold):
                r=255
                g=255
                b=255
                image.putpixel((i,j),(r,g,b))
            else:
                r = 0
                g = 0
                b = 0
                image.putpixel((i, j), (r, g, b))

    # 灰度图片
    image = image.convert('L')
    return image

# 文本转向量
def text2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        idx = CHAR_SET.index(c)
        vector[i][idx] = 1.0
    return vector

# 向量转回文本
def vec2text(vec):
    text = []
    for i, c in enumerate(vec):
        text.append(CHAR_SET[c])
    return "".join(text)

"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""

# 生成一个训练batch
def get_next_batch(batch_size=128):
    # 输入的图片为160 * 60的，灰度化预处理以后为一维数组，每张图片总共有9600个输入值
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 输出的字符集有62个字符，并且每张图片有4位字符，总共有4 * 62 = 248个输出值
    batch_y = np.zeros([batch_size, MAX_CAPTCHA, CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = tf.reshape(otsu(text, image), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        #image = tf.reshape(np.array(denoising(image)), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

        batch_x[i, :] = image
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

# 定义CNN
def crack_captcha_cnn():
    '''
    输入层有9600个值，输出层有248个值，如果使用全连接层作为隐藏层则会需要天量的计算

    所以需要先使用卷积核池化操作尽可能的减少计算量（如果有一些深度学习基础的同学应该知道计算机视觉中一般都是用卷积升级网络来解决这类问题）

    图片像素不高，所以使用的卷积核和池大小不能太大，优先考虑3 * 3 和5 * 5 的卷积核，池大小使用2 * 2

    按照下面的神经网络模型，卷积池化以后的输出应该是128 * 17 * 5 = 10880（如果最后一层的深度仍然使用64的话，大小会减为一半）
    '''
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    #model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu))
    #model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu))
    #model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'))

    # 输出的的每一位的字符之间没有关联关系，所以仍然将输出值看成4组，需要将输出值调整为(4, 62)的数组
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN))
    model.add(tf.keras.layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN]))

    #识别的原理是计算每一位字符上某个字符出现的可能性最大，所以每张图片都是一个4位的多分类问题，最终输出使用softmax进行归一化
    model.add(tf.keras.layers.Softmax())

    return model

# 训练
def train():
    try:
        model = tf.keras.models.load_model(SAVE_PATH)
    except Exception as e:
        print('#######Exception', e)
        model = crack_captcha_cnn()

    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')

    for times in range(500000):
        batch_x, batch_y = get_next_batch(512)
        print('times=', times, ' batch_x.shape=', batch_x.shape, ' batch_y.shape=', batch_y.shape)
        model.fit(batch_x, batch_y, epochs=4)
        print("y预测=\n", np.argmax(model.predict(batch_x), axis=2))
        print("y实际=\n", np.argmax(batch_y, axis=2))

        if 0 == times % 10:
            print("save model at times=", times)
            model.save(SAVE_PATH)

def predict():
    model = tf.keras.models.load_model(SAVE_PATH)
    success = 0
    count = 100
    for _ in range(count):
        data_x, data_y = get_next_batch(1)
        prediction_value = model.predict(data_x)
        data_y = vec2text(np.argmax(data_y, axis=2)[0])
        prediction_value = vec2text(np.argmax(prediction_value, axis=2)[0])

        if data_y.upper() == prediction_value.upper():
            print("y预测=", prediction_value, "y实际=", data_y, "预测成功。")
            success += 1
        else:
            print("y预测=", prediction_value, "y实际=", data_y, "预测失败。")
        print("预测", count, "次", "成功率=", success / count)

if __name__ == '__main__':
    train()
    predict()