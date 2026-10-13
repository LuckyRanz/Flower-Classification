"""项目参数集中在这个文件里。

改实验，只动这里；别去改代码。这是真实项目的习惯——
三个月后你回头看，能一眼说出"当时用的什么设置"。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # 数据集落地的地方
OUT_DIR = ROOT / "outputs"        # 模型、日志、图，都写这里
OUT_DIR.mkdir(exist_ok=True)

# ======================== 数据 ========================
# 原图普遍在 500x500 左右，CPU 上跑不动。缩到多少，是速度和质量的第一笔交易。
IMG_SIZE = 96

# 102 类全上，CPU 上每轮要几分钟；先用前 N 类把流程跑通。
# 流程对了之后，把这个数字改成 102 就能跑全量——代码不用动。
NUM_CLASSES = 20

BATCH_SIZE = 32
VAL_RATIO = 0.2                   # 从训练池里切多少当验证集
SEED = 0

# 读图的并行进程数。这个值我实测过，别凭感觉改：
#   num_workers=0 → 读一轮 1.2 秒，没有额外开销
#   num_workers=4 → 读一轮 0.4 秒，但启动时要等 24 秒
# 原因是 Windows 上每开一个 worker 都要重新 import 一次 torch（6~12 秒/个），
# 而这个规模的数据单进程读一轮本来就只有 1 秒多。所以【0 反而最快】。
# 等数据量涨到几千张（比如把 NUM_CLASSES 改成 102）时，再改成 4 更划算——
# data_utils.py 会自动帮你打开 persistent_workers，让 worker 只启动一次。
NUM_WORKERS = 0

# ======================== 训练 ========================
EPOCHS = 12
LR = 1e-3                         # 从零训练用的学习率

# 微调预训练模型的学习率。这个值不是"沿用惯例觉得小一点更稳"——是我实测出来的。
# 反直觉的地方：用了预训练，学习率反而【不比从零训练小】。
# 因为主干被冻住了，真正在学的是那个【随机初始化的新分类头】，
# 它一开始啥也不会，需要大步子才走得动。
# 实测（同一份数据、12 轮）：
#     lr=1e-4 → 验证 26.2%      lr=1e-3 → 验证 86.2%      lr=1e-2 → 验证 82.5%
# 1e-4 不是"更稳"，是"根本没开始走"。
# 等你要解冻主干、真的去微调那些预训练参数时，才该把它调小。
LR_FINETUNE = 1e-3
WEIGHT_DECAY = 1e-4

# ======================== 模式 ========================
# "scratch"  ：自己搭的小 CNN，从零训
# "transfer" ：ImageNet 预训练的 resnet18，只训最后那层分类头
# 这两个名字，你会在 train.py 里看到它们怎么分岔。
MODE = "scratch"

# ======================== 复现 ========================
def seed_everything(seed: int = SEED):
    """把随机性按住。第 0 步跑之前先调用它，结果才可复现。"""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def describe() -> str:
    """把当前配置打印成一行，方便记实验日志。"""
    return (f"mode={MODE} img={IMG_SIZE} classes={NUM_CLASSES} "
            f"epochs={EPOCHS} lr={LR if MODE == 'scratch' else LR_FINETUNE} "
            f"bs={BATCH_SIZE} seed={SEED}")
