# import datetime
# import os
import sys
import argparse
import logging
import cv2
import torch
# import torch.utils.data
# import torch.optim as optim
# import torch.nn as nn
# from tqdm import tqdm
# from torchsummary import summary
from PIL import Image
import numpy as np

from torchvision import transforms
from models import get_network

import matplotlib.pyplot as plt
import time

logging.basicConfig(level=logging.INFO)
CURRENT_FILE = sys.argv[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference slipknotnet")

    # Network
    parser.add_argument(
        "--network", type=str, default="slipknotnet", help="Network Name in .models"
    )

    args = parser.parse_args()
    return args


def resize_image(image, width=1920, height=1080, resized_used=True):
    if resized_used:
        new_size = (width, height)
        resized_image = cv2.resize(image, new_size)
        return resized_image
    else:
        return image


def post_process(output):
    output = output.squeeze().detach().cpu().numpy()

    output[output < 0] = 0.0

    return output


def show_image_and_mask(image, mask, normalize=True):
    """
    显示 RGB 图像和掩码。

    Args:
        image (Tensor): 输入的 RGB 图像，形状为 [C, H, W] 或 [B, C, H, W]。
        mask (Tensor): 掩码图像，形状为 [1, H, W] 或 [B, 1, H, W]。
        normalize (bool): 是否标准化图像（将像素值归一化到 0-1 范围内）。
    """
    # 如果输入是批量数据，选择第一张图像
    if image.dim() == 4:
        image = image[0]
        # mask = mask[0]

    # 将图像从 Tensor 转换为 numpy 数组，并转换为 [H, W, C] 格式
    image = image.permute(1, 2, 0).cpu().numpy()

    # 如果需要，可以进行归一化
    if normalize:
        image = np.clip(image, 0, 1)

    # 将掩码从 Tensor 转换为 numpy 数组
    # mask = mask.squeeze().cpu().numpy()
    # mask = mask.squeeze().detach().cpu().numpy()
    # mask = np.clip(mask, 0, 1)

    mask = np.uint8(resize_image(mask, raw_w, raw_h) * 255)

    # cmap = cm.get_cmap('gray')
    # mask = cmap(mask)[:, :, 0]

    print(mask.shape)
    # _, mask = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./results/mask_pred.png", mask)

    # 创建一个图形，显示图像和掩码
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(resize_image(image, raw_w, raw_h))
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    # axes[1].imshow(mask)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.show()


if __name__ == "__main__":
    args = parse_args()

    model_path = "checkpoint/2025-02-26_01-46-57_sutureresnet_/epoch_50" 
    model_path = "checkpoint/2025-08-21_23-43-20_slipknotnet_/epoch_02" 

    print("model path: ", model_path)
    input_channels = 3
    slipknotnet = get_network(args.network)
    # print("input_channels: ", input_channels)
    net = slipknotnet(input_channels=input_channels)

    net.load_state_dict(torch.load(model_path))

    device = torch.device("cuda:0")
    net = net.to(device)
    net.eval()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")

    image_path = "demo/davinci_1.png"
    # image_path = 'demo/davinci_2.png'

    image_raw = Image.open(image_path).convert("RGB")
    raw_w, raw_h = image_raw.size
    transform = transforms.Compose(
        [
            transforms.Resize((800, 800)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为 Tensor 格式
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet均值和标准差
        ]
    )

    transform0 = transforms.Compose(
        [
            transforms.Resize((800, 800)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为 Tensor 格式
        ]
    )
    start_time = time.time()

    tensor_image_raw = transform0(image_raw)
    tensor_image_raw = tensor_image_raw.unsqueeze(0)

    image = transform(image_raw)
    image = image.unsqueeze(0)

    xc = image.to(device)
    output = net(xc)
    output = post_process(output)
    end_time = time.time()

    print("time cost:", end_time - start_time)

    show_image_and_mask(tensor_image_raw, output)
