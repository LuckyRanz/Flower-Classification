"""两个模型，代表图像分类的两条路。

scratch  —— 自己搭一个小卷积网络，从零训练。能看到细节，但很难学好。
transfer —— 借用在 ImageNet 上已经训好的 resnet18，只换最后一层。

真实项目里 90% 的情况走第二条。但你得先看过第一条的失败，
才知道第二条为什么值钱。
"""

import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """从零搭的卷积网络。输入 (B, 3, S, S)，输出 (B, num_classes)。

    结构上就是三个 block，每个 block 做同一件事：
        Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool
    每过一个 block，空间尺寸减半（MaxPool 干的），通道数翻倍（Conv 干的）。
    最后用全局平均池化把特征图压成一个向量，交给全连接层分类。
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),       # 96 -> 48
            block(32, 64),      # 48 -> 24
            block(64, 128),     # 24 -> 12
        )
        self.pool = nn.AdaptiveAvgPool2d(1)     # 无论输入多大，都压成 1x1 自适应平均池化层
        self.classifier = nn.Sequential(
            nn.Flatten(),    # 把特征整理成向量
            nn.Dropout(0.3),     # 训练时随机屏蔽部分特征
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_scratch_model(num_classes: int = 20) -> nn.Module:
    return SmallCNN(num_classes=num_classes)


def build_transfer_model(num_classes: int = 20, freeze_backbone: bool = True) -> nn.Module:
    """迁移学习。

    第一次运行会下载 resnet18 的 ImageNet 权重（约 45 MB），之后走本地缓存。

    freeze_backbone=True 时，主干所有参数的 requires_grad 被关掉——
    意思是"这些眼睛已经够好了，别动它们，我只训最后那层分类头"。
    这是 CPU 上唯一跑得动、又快又准的做法。
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # 原模型输出 1000 类（ImageNet），换成本项目的类别数。
    # 新加的这层默认 requires_grad=True，所以只有它在学。
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def count_params(model: nn.Module) -> tuple:
    """返回 (总参数量, 可训练参数量)，单位：百万。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(total / 1e6, 2), round(trainable / 1e6, 2)


def lock_frozen_batchnorm(model: nn.Module) -> int:
    """把"已经被冻住"的 BatchNorm 层按回 eval 模式，返回锁了几层。

    这是迁移学习里一个很容易忽略的坑。BatchNorm 在训练模式下会做两件事：
    用当前 batch 的统计量做归一化，并且更新自己记住的 running_mean / running_var。

    主干是从 ImageNet 上学来的，它的 running stats 是拿上千万张图喂出来的。
    你现在拿几百张花卉图去覆盖它，等于把人家调好的刻度擦掉重写——
    表现就是"明明用了预训练，效果还不如随便训"。

    所以规矩是：参数被冻结的层，统计量也一起冻住。
    代码里靠 requires_grad 来判断，正好和 freeze_backbone 这件事对齐。
    """
    locked = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and m.weight is not None \
                and not m.weight.requires_grad:
            m.eval()
            locked += 1
    return locked


def build_model(mode: str, num_classes: int) -> nn.Module:
    if mode == "scratch":
        return build_scratch_model(num_classes)
    if mode == "transfer":
        return build_transfer_model(num_classes, freeze_backbone=True)
    raise ValueError(f"不认识 mode={mode!r}，只能是 'scratch' 或 'transfer'")


if __name__ == "__main__":
    # 跑一下，看两个模型各有多少参数——这是它们能力差距的第一手证据。
    for mode in ["scratch", "transfer"]:
        m = build_model(mode, 20)
        total, trainable = count_params(m)
        print(f"  {mode:<9} 总参数 {total:>6.2f} M   可训练 {trainable:>6.2f} M")
