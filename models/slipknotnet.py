import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),  # 恢复通道
            nn.Sigmoid(),  # 激活函数，输出权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 全局平均池化并展平
        y = self.fc(y).view(b, c, 1, 1)  # 全连接层并恢复形状
        return x * y  # 将权重应用到原始特征图


# Double Convolution Block for UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = DoubleConv(out_channels * 2, out_channels)
            self.se = SEModule(out_channels)  # 添加 SE 模块

        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
            self.se = SEModule(out_channels)  # 添加 SE 模块

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        x1 = F.relu(self.conv1(x1))
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x_cat = torch.cat((x2, x1), dim=1)
        x_cat = self.conv2(x_cat)
        x_cat = self.se(x_cat)  # 应用 SE 模块
        return F.relu(x_cat)


class SutruePre(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(SutruePre, self).__init__()
        self.conv1 = DoubleConv(in_channels, 16)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, output_channels, kernel_size=2, padding=0)
        # self.conv3 = nn.Conv2d(8, output_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(p=0.1)
        # self.resize = nn.UpsamplingBilinear2d(size=(336, 336))
        # self.resize = nn.UpsamplingBilinear2d(size=(1079, 1517))
        self.resize = nn.UpsamplingBilinear2d(size=(800, 800))

    def forward(self, x):
        # x = F.dropout(self.conv1(x), p=0.5, training=self.training)
        x = self.dropout(self.conv1(x))
        x = F.relu(self.conv2(x))
        # return self.conv3(x)
        return self.resize(self.conv3(x))


class UNetDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNetDecoder, self).__init__()

        self.up1 = Up(input_channels, 1024)
        self.up2 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 64)
        self.final_conv = SutruePre(64, output_channels)

    def forward(self, x, enc_features):
        # print('before decoder:', x.shape)
        d1 = self.up1(x, enc_features[3])
        d2 = self.up2(d1, enc_features[2])
        d3 = self.up3(d2, enc_features[1])
        d4 = self.up4(d3, enc_features[0])

        # print('d1:', d1.shape)
        # print('d2:', d2.shape)
        # print('d3:', d3.shape)
        # print('d4:', d4.shape)

        out = self.final_conv(d4)
        # print('out:', out.shape)
        return out


# Final model: combining ResNet50 encoder and UNet decoder
class ResNet50UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(ResNet50UNet, self).__init__()

        # Extract intermediate features from the ResNet50 encoder
        resnet50 = models.resnet50(pretrained=True)

        # 冻结所有层
        for param in resnet50.parameters():
            param.requires_grad = False

        # # 解冻最后几层卷积层和全连接层
        # for name, param in resnet50.named_parameters():
        #     if 'layer4' in name:  # 解冻 ResNet50 的最后一个残差块（layer4）
        #         param.requires_grad = True
        #     elif 'fc' in name:  # 解冻全连接层
        #         param.requires_grad = True

        self.enc1 = nn.Sequential(*list(resnet50.children())[:3])  # Layer1
        self.enc2 = nn.Sequential(*list(resnet50.children())[3:5])  # Layer2
        self.enc3 = nn.Sequential(*list(resnet50.children())[5])  # Layer3
        self.enc4 = nn.Sequential(*list(resnet50.children())[6])  # Layer4

        # self.decoder = UNetDecoder(input_channels=2048, output_channels=output_channels)
        self.decoder = UNetDecoder(input_channels=1024, output_channels=output_channels)

        # for name, param in self.enc4.named_parameters():
        #     param.requires_grad = True

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        # print('before encoder:', x.shape)
        # print('e1:', e1.shape)
        # print('e2:', e2.shape)
        # print('e3:', e3.shape)
        # print('e4:', e4.shape)

        # Decoder
        out = self.decoder(e4, [e1, e2, e3, e4])
        return out

    def compute_loss(self, xc, yc, alpha=0.5):
        suture_pred = self(xc)

        loss_dice = dice_loss(suture_pred, yc)
        loss_focal = focal_loss(suture_pred, yc)
        all_loss = alpha * loss_dice + (1 - alpha) * loss_focal

        return {
            "loss": all_loss,
            "losses": {
                # "l1_loss": loss_l1,
                "focal_loss": loss_focal,
                "dice_loss": loss_dice,
            },
            "pred": {
                "suture_prediction": suture_pred,
            },
        }


def focal_loss(pred, target, alpha=0.25, gamma=2):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # 确保 pred 在 [0, 1] 范围内
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# Example usage
if __name__ == "__main__":
    model = ResNet50UNet(output_channels=1)  # Regression output has 1 channel
    # # print(model)
    input_image = torch.randn(
        1, 3, 800, 800
    )  # Batch size 1, 3-channel RGB image, 256x256 resolution
    output = model(input_image)
    print(output.shape)  # Should output (1, 1, 256, 256)

    # # 创建一个简单的测试用例
    # pred = torch.tensor([[0.8, 0.2], [0.6, 0.4]])  # 模型输出
    # target = torch.tensor([[1, 0], [1, 0]])  # 目标掩码

    # # 计算 Dice Loss
    # loss = weighted_dice_loss(pred, target)
    # print("Dice Loss:", loss.item())  # 应该输出一个合理的值（非 0.5）
