"""数据管线：从磁盘上的花卉照片，到可以喂给模型的张量。

一句话概括这条流水线：
    文件 → PIL 图 → 【变换】→ 张量 → DataLoader 打成一包一包
中间那个【变换】是本文件的主角，它决定了模型"看到"什么样的图。
"""

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import Flowers102

# ImageNet 的均值和标准差。为什么用别人的数字？
# 因为预训练的主干就是在这套刻度上学的，你得跟它保持一致，
# 否则在它眼里你的图是"另一种颜色"。（从零训练时用它们也无害。）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _labels_of(ds) -> list:
    """取数据集里每张图的标签。

    torchvision 把标签缓存在私有属性里，直接读比一张张打开图片快无数倍。
    不同版本属性名不一样，所以挨个试；实在没有才退回去逐张读图。
    """
    for attr in ("_labels", "labels", "targets"):
        if hasattr(ds, attr):
            return list(getattr(ds, attr))
    return [ds[i][1] for i in range(len(ds))]


class _WithTransform(Dataset):
    """给一个现成的 Dataset 换一套 transform。

    ConcatDataset / Subset 都不能单独改 transform，所以包一层。
    训练集要随机增强、验证集不能动，必须能分开设，这个包装器就是干这个的。
    """

    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        return self.transform(img), y


def build_transforms(img_size: int, train: bool):
    """训练用和评估用的两套变换。

    区别只有一处：训练那套带【随机】增强，评估那套是确定性的。
    为什么评估不能加随机？——同一张图两次评分必须一样，
    否则误差棒比你要比较的差距还大。
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),        # 左右翻转：花不会因为翻面变成别的花
            transforms.RandomRotation(15),            # 小角度旋转
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_datasets(data_dir, img_size: int, num_classes: int,
                   val_ratio: float = 0.2, seed: int = 0,
                   test_per_class: int = 15):
    """返回 (train_ds, val_ds, test_ds)。

    关于划分，这里做了一个不那么"教科书"的决定，说明一下：

    Flowers102 官方把数据分成 train(1020) / val(1020) / test(6149)——
    每类在 train 里只有 10 张。只拿 train 训，每类 10 张图实在太少，
    连"数据不够"这件事本身都会被别的因素盖住。

    所以这里把 train + val 合起来当训练池（每类 20 张），再按 val_ratio
    切出验证集；官方的 test 留作最终测试。目的很单纯：让"能不能学好"
    这个信号干净一点。
    """
    root = str(data_dir)

    raw_train = Flowers102(root, split="train", download=False)
    raw_val = Flowers102(root, split="val", download=False)
    raw_test = Flowers102(root, split="test", download=False)

    pool = ConcatDataset([raw_train, raw_val])
    pool_labels = _labels_of(raw_train) + _labels_of(raw_val)
    keep = [i for i, y in enumerate(pool_labels) if y < num_classes]

    pool_kept = Subset(pool, keep)
    n_val = int(len(pool_kept) * val_ratio)
    n_train = len(pool_kept) - n_val
    g = torch.Generator().manual_seed(seed)
    tr_part, va_part = random_split(range(len(pool_kept)), [n_train, n_val], generator=g)

    train_ds = _WithTransform(Subset(pool_kept, tr_part.indices),
                              build_transforms(img_size, train=True))
    val_ds = _WithTransform(Subset(pool_kept, va_part.indices),
                            build_transforms(img_size, train=False))

    # 测试集：官方 test 里每类抽 test_per_class 张，免得评估一次要跑上千张图
    test_labels = _labels_of(raw_test)
    picked, per_class = [], {}
    for i, y in enumerate(test_labels):
        if y < num_classes and per_class.get(y, 0) < test_per_class:
            picked.append(i)
            per_class[y] = per_class.get(y, 0) + 1
    test_ds = _WithTransform(Subset(raw_test, picked),
                             build_transforms(img_size, train=False))

    return train_ds, val_ds, test_ds


def build_loaders(cfg):
    """按 config 组装三个 DataLoader。"""
    train_ds, val_ds, test_ds = build_datasets(
        cfg.DATA_DIR, cfg.IMG_SIZE, cfg.NUM_CLASSES,
        val_ratio=cfg.VAL_RATIO, seed=cfg.SEED)

    common = dict(num_workers=cfg.NUM_WORKERS, pin_memory=False)
    if cfg.NUM_WORKERS > 0:
        # 不打开这个，每个 epoch 结束时 worker 会被销毁、下一个 epoch 再重建，
        # Windows 上每次重建都要重新 import torch，白白浪费十几秒。
        common["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE,
                             shuffle=False, **common)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import config as cfg

    tr, va, te = build_datasets(cfg.DATA_DIR, cfg.IMG_SIZE, cfg.NUM_CLASSES,
                                val_ratio=cfg.VAL_RATIO, seed=cfg.SEED)
    print(f"  训练 {len(tr):>4} 张")
    print(f"  验证 {len(va):>4} 张")
    print(f"  测试 {len(te):>4} 张")

    x, y = tr[0]
    print(f"\n  一张图变成张量后：shape={tuple(x.shape)}  dtype={x.dtype}")
    print(f"  取值被归一化到 [{x.min():.2f}, {x.max():.2f}]")
    print(f"  标签 {y}（0~{cfg.NUM_CLASSES - 1}）")

    # 训练集带随机增强，所以同一张图取两次结果不同——这正是我们要的
    a, _ = tr[0]
    b, _ = tr[0]
    print(f"\n  训练集同一张图取两次是否相同：{torch.allclose(a, b)}"
          f"   ← 不同才对，因为训练集带随机增强")
    c, _ = va[0]
    d, _ = va[0]
    print(f"  验证集同一张图取两次是否相同：{torch.allclose(c, d)}"
          f"   ← 必须相同，评估不能有随机性")
