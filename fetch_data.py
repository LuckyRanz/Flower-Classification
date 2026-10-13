# 把 Oxford Flowers102 数据集下载到项目目录，并做一次体检
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("A. 依赖检查（Flowers102 的标签存在 .mat 文件里，需要 scipy）")
print("=" * 60)
try:
    import scipy
    print(f"  scipy {scipy.__version__} ✔")
except ImportError:
    print("  scipy 没装 —— 需要 pip install scipy")

print()
print("=" * 60)
print("B. 下载 Flowers102（三个文件，合计约 330 MB）")
print("=" * 60)

from torchvision.datasets import Flowers102

for split in ["train", "val", "test"]:
    ds = Flowers102(root=str(DATA_ROOT), split=split, download=True)
    print(f"  {split:<6} {len(ds):>5} 张")

print()
print("=" * 60)
print("C. 体检：看一眼真实样本和类别分布")
print("=" * 60)
from collections import Counter

train = Flowers102(root=str(DATA_ROOT), split="train", download=False)
img, label = train[0]
print(f"  单张图：PIL 对象 {img}")
print(f"    尺寸 {img.size}   模式 {img.mode}")
print(f"    标签 {label}（0~101 的整数）")
print(f"    类别名 {train.classes[label] if hasattr(train, 'classes') else '(torchvision 不直接给名字)'}")
print()

counts = Counter(train._labels)
print(f"  训练集 {len(train)} 张，{len(counts)} 个类别")
print(f"    每类样本数：最少 {min(counts.values())}，最多 {max(counts.values())}")
print(f"    → 102 类分 1020 张训练图，平均每类只有 {len(train) / 102:.1f} 张")
print()
print("  这个数字很重要：每类 10 张图，从零训 CNN 几乎不可能学好。")
print("  真实工作里遇到这种规模，标准答案是【迁移学习】——第 3 步会走到。")
