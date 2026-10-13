# 侦查：本机跑图像任务的能力盘点
import os
import platform
import sys
import multiprocessing

print("=" * 60)
print("A. Python 与深度学习库")
print("=" * 60)
print(f"  Python {sys.version.split()[0]}  ({platform.machine()})")

try:
    import torch
    print(f"  torch {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    print(f"  MPS 可用 : {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
except Exception as e:
    print(f"  torch 导入失败: {e}")

try:
    import torchvision
    print(f"  torchvision {torchvision.__version__}")
except Exception as e:
    print(f"  torchvision 导入失败: {type(e).__name__}: {e}")

try:
    from PIL import Image
    print(f"  Pillow {Image.__version__ if hasattr(Image, '__version__') else 'ok'}")
except Exception as e:
    print(f"  Pillow 导入失败: {e}")

print()
print("=" * 60)
print("B. CPU 资源（决定能跑多大的图、多大的 batch）")
print("=" * 60)
print(f"  逻辑核心数: {multiprocessing.cpu_count()}")

try:
    import psutil
    print(f"  物理核心数: {psutil.cpu_count(logical=False)}")
    vm = psutil.virtual_memory()
    print(f"  内存: 总 {vm.total / 1024**3:.1f} GB，可用 {vm.available / 1024**3:.1f} GB")
except Exception:
    print("  （psutil 没装，跳过内存信息）")

print()
print("=" * 60)
print("C. 一次前向+反向 的耗时标定（决定训练可行性）")
print("=" * 60)
import time
import torch
import torch.nn as nn

for size in [64, 96, 128]:
    x = torch.randn(32, 3, size, size)
    m = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 5),
    )
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    t0 = time.time()
    for _ in range(3):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(x), torch.randint(0, 5, (32,)))
        loss.backward()
        opt.step()
    dt = (time.time() - t0) / 3
    print(f"  {size}x{size}, batch=32 → 每步 {dt * 1000:.0f} ms"
          f"   跑 1 万步约 {dt * 10000 / 60:.1f} 分钟")

print()
print("=" * 60)
print("D. torchvision 里的现成数据集（不需要下载的）")
print("=" * 60)
import torchvision.models as tvm
print("  可用预训练主干（CPU 上做迁移学习会很快）:")
for name in ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0", "vgg16"]:
    print(f"    - {name}")
print()
print("  权重默认缓存在:", os.path.expanduser("~/.cache/torch/hub/checkpoints"))
cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
if os.path.isdir(cache):
    print("  已有:", os.listdir(cache))
else:
    print("  该目录还不存在（说明还没下过预训练权重）")
