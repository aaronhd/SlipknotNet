import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import random

try:
    from .illumination import random_illumination_augmentation
except ImportError:
    from illumination import random_illumination_augmentation


class ImageMaskDataset(Dataset):
    def __init__(self, data_path, start=0.0, end=1.0, image_list=None, light_aug=True):
        """
        Args:
            image_paths (list of str): 图像文件路径列表。
            mask_paths (list of str): 掩码文件路径列表。
            transform (callable, optional): 可选的图像预处理操作。
            mask_transform (callable, optional): 可选的掩码预处理操作。
        """
        print("Davinci-sliputure_Dataset")
        self.light_aug = light_aug
        self.data_path = data_path
        if image_list is None:
            print("used default image list")
            image_list = glob.glob(os.path.join(data_path, "input_data", "*.jpg"))

        self.used_image_list = image_list[
            int(len(image_list) * start) : int(len(image_list) * end)
        ]
        print("data loader length: ", len(self.used_image_list))
        # self.image_paths = image_paths
        # self.mask_paths = mask_paths
        # self.transform = transform
        # self.mask_transform = mask_transform

        # 定义图像和掩码的预处理操作
        self.transform = transforms.Compose(
            [
                transforms.Resize((800, 800)),  # 调整图像大小
                transforms.ToTensor(),  # 转换为 Tensor 格式
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet均值和标准差
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((800, 800)),  # 调整掩码大小
                transforms.ToTensor(),  # 转换为 Tensor 格式
            ]
        )

    def __getitem__(self, idx):
        # 加载图像和对应的掩码
        image_path = self.used_image_list[idx]

        local_image_name = image_path.split("/")[-1][:-4]

        mask_path = os.path.join(
            self.data_path, "groundtruth", "mask", local_image_name + "_mask.png"
        )

        image = Image.open(image_path).convert("RGB")  # 加载 RGB 图像
        mask = Image.open(mask_path).convert("L")  # 加载灰度掩码图
        # image = rotate_image(image)
        # mask = rotate_image(mask)

        if self.light_aug:
            image = random_illumination_augmentation(image)

        # 应用预处理（如果有）
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.used_image_list)


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
        mask = mask[0]

    # 将图像从 Tensor 转换为 numpy 数组，并转换为 [H, W, C] 格式
    image = image.permute(1, 2, 0).cpu().numpy()

    # 如果需要，可以进行归一化
    if normalize:
        image = np.clip(image, 0, 1)

    # 将掩码从 Tensor 转换为 numpy 数组
    mask = mask.squeeze().cpu().numpy()

    # 创建一个图形，显示图像和掩码
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.show()


def rotate_image(image, max_angle=30, fill_color=0):
    # 随机选择一个旋转角度，范围为 -max_angle 到 max_angle
    angle = random.uniform(-max_angle, max_angle)
    rotated_image = image.rotate(
        angle, expand=True, fillcolor=fill_color
    )  # expand=True 保证旋转后图像不被裁剪
    return rotated_image


if __name__ == "__main__":
    # data_path = "/media/aaronsamd37/hard_1/cao/Davinci-sliputure_Dataset"
    data_path = os.environ["SlipknotNet_FOLDER"] + "/data/Davinci-sliputure_Dataset"
    print(data_path)
    # 创建数据集和数据加载器
    dataset = ImageMaskDataset(data_path, start=0.0, end=1.0, light_aug=True)
    # dataloader = DataLoader(
    #     dataset, batch_size=1, shuffle=True
    # )
    id = random.randint(0, len(dataset))
    print(id)
    image, mask = dataset.__getitem__(id)
    show_image_and_mask(image, mask)
