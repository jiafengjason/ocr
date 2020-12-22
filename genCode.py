#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import os
import shutil
import random
import time
import math
import string
import argparse
from captcha.image import ImageCaptcha

ARGS = None
#用于生成验证码的字符集
CHAR_SET = []

#验证码的长度，每个验证码由4个数字组成
CAPTCHA_LEN = 4
 
#验证码图片的存放路径
CAPTCHA_IMAGE_PATH = 'images/'
FONTS_PATH = 'fonts/'
#用于模型测试的验证码图片的存放路径，它里面的验证码图片作为测试集
TEST_IMAGE_PATH = 'test/'
 
#生成验证码图片，4位的十进制数字可以有10000种验证码
def genAllImage():
    count = 0
    total = int(math.pow(len(CHAR_SET),CAPTCHA_LEN))

    fonts = []
    for fontName in os.listdir(FONTS_PATH):
        fontPath = os.path.join(FONTS_PATH,fontName)
        fonts.append(fontPath)

    for captcha_text in range(total):
        captcha_text = str(captcha_text).zfill(4)
        image = ImageCaptcha(fonts = fonts)
        #image = ImageCaptcha(width=width, height=height, fonts=['/path/to/A.ttf', '/path/to/B.ttf']) 修改宽、高和字体
        #图片格式改成其他可能会有问题
        image.write(captcha_text, CAPTCHA_IMAGE_PATH + captcha_text + '.png')
        count += 1
        sys.stdout.write("\rCreating %d/%d" % (count, total))
        sys.stdout.flush()

#从验证码的图片集中取出一部分作为测试集，这些图片不参加训练，只用于模型的测试
def genRandomImage():
    count = 0
    '''
    fileNameList = []
    for filePath in os.listdir(CAPTCHA_IMAGE_PATH):
        captcha_name = filePath.split(os.path.sep)[-1]
        fileNameList.append(captcha_name)
    '''
    fonts = []
    for fontName in os.listdir(FONTS_PATH):
        fontPath = os.path.join(FONTS_PATH,fontName)
        fonts.append(fontPath)

    random.seed(time.time())
    #random.shuffle(fileNameList)
    for i in range(ARGS.num):
        #name = fileNameList[i]
        captcha_text = ''.join([random.choice(CHAR_SET) for j in range(4)])
        #shutil.move(CAPTCHA_IMAGE_PATH + name, TEST_IMAGE_PATH + name)
        image = ImageCaptcha(fonts = fonts)
        image.write(captcha_text, CAPTCHA_IMAGE_PATH + captcha_text + '.png')
        count += 1
        sys.stdout.write("\rCreating %d/%d" % (count, ARGS.num))
        sys.stdout.flush()

if __name__ == '__main__':
    if os.path.exists(CAPTCHA_IMAGE_PATH):
        shutil.rmtree(CAPTCHA_IMAGE_PATH,True)
    os.makedirs(CAPTCHA_IMAGE_PATH)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--function", dest = 'fun')
    parser.add_argument("-m","--mode", choices = ['all', 'random'], dest = 'mode', default='random')
    parser.add_argument("-s","--set", choices = ['all', 'digit', 'lower', 'upper', 'letters'], dest = 'set', default='digit')
    parser.add_argument("-n","--num", dest = 'num', type = int, default=100)
    ARGS = parser.parse_args()
    print(ARGS)
    
    if ARGS.set == 'all':
        CHAR_SET = list(string.digits + string.ascii_lowercase + string.ascii_uppercase)
    elif ARGS.set == 'digit':
        CHAR_SET = list(string.digits)
    elif ARGS.set == 'lower':
        CHAR_SET = list(string.ascii_lowercase)
    elif ARGS.set == 'upper':
        CHAR_SET = list(string.ascii_uppercase)
    elif ARGS.set == 'letters':
        CHAR_SET = list(string.ascii_letters)

    if ARGS.mode == 'all':
        genAllImage()
    elif ARGS.mode == 'random':
        genRandomImage()
    sys.stdout.write("\nFinished")
    sys.stdout.flush()