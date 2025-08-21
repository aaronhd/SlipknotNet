import numpy as np
from PIL import Image, ImageEnhance
import random


def adjust_brightness(img, delta):
    """调整亮度: delta范围 [-1, 1]，例如0.2表示亮度增加20%"""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1 + delta)


def adjust_contrast(img, factor):
    """调整对比度: factor>1增强对比度，<1减弱对比度"""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def adjust_gamma(img, gamma):
    """伽马校正: gamma=1.0为原图，<1变亮，>1变暗"""
    if gamma <= 0:
        gamma = 0.1
    table = [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
    table = np.array(table).astype(np.uint8)
    return img.point(table)


def add_gaussian_noise(img, mean=0, var=30):
    """添加高斯噪声"""
    img = np.array(img)  # 转换为NumPy数组
    row, col, ch = img.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def random_illumination_augmentation(img):
    # print('light aug used')
    # 随机亮度调整（-30%到+30%）
    if random.random() < 0.75:
        delta = random.uniform(-0.3, 0.3)
        img = adjust_brightness(img, delta)

    # 随机对比度调整（0.7到1.3倍）
    if random.random() < 0.75:
        factor = random.uniform(0.7, 1.3)
        img = adjust_contrast(img, factor)

    # # 随机伽马调整（0.8到1.2）
    # if random.random() < 0.5:
    #     gamma = random.uniform(0.8, 1.2)
    #     img = adjust_gamma(img, gamma)

    # 随机高斯噪声
    if random.random() < 0.25:
        var = random.uniform(10, 50)
        img = add_gaussian_noise(img, var=var)

    return img
