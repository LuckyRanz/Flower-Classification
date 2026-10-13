"""花卉项目的概念预备 —— 这不是一节课，是一把尺子。

你说想实战，不想学新东西。我尊重这个决定，所以这份文件不是作业，
你也不用背。它只解决一件事：**当你在项目里撞到一个没见过的词、
心里发毛的时候，能有个地方立刻看到它到底在干什么。**

每个部分都很短（跑完总共一分多钟），而且都配了一个"能看出差别"的小实验——
不是听我描述概念，是让程序把差别打出来给你看。

    用法：
        python concepts_primer.py            # 从头看到尾
        python concepts_primer.py 2          # 只看第 2 部分

六个部分，正好对上 README 里那些新词：
    1. 一张图在程序里到底是什么
    2. 卷积核：一次算给你看（从 9 个数字手算起）
    3. 池化与通道：为什么越往里越抽象
    4. 迁移学习为什么能成立（这是本项目的心脏）
    5. 数据增强到底改了什么
    6. 过拟合在图像任务里长什么样
"""

import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import Flowers102

import config as cfg
from data_utils import build_transforms
from model import SmallCNN

# 这些小实验都是单进程跑的，别把 16 个核全占了
torch.set_num_threads(4)


def title(n, text):
    print()
    print("=" * 66)
    print(f"  第 {n} 部分 · {text}")
    print("=" * 66)


def labels_of(ds) -> list:
    """取数据集的标签列表。torchvision 缓存在私有属性里，读它比开图快。"""
    for attr in ("_labels", "labels", "targets"):
        if hasattr(ds, attr):
            return list(getattr(ds, attr))
    return [ds[i][1] for i in range(len(ds))]


def show(m):
    """把一个 2D 矩阵按数值大小打印成点阵，用来看出图案。"""
    chars = " .:-=+*#%@"
    lo, hi = float(m.min()), float(m.max())
    span = (hi - lo) or 1.0
    for row in m:
        line = "".join(chars[min(9, int((float(v) - lo) / span * 9.999))] for v in row)
        print("      |" + line + "|")


def show_signed(m):
    """带正负号的点阵图。0 画成空白，数值越大画得越满，正负都用同一套档位。"""
    chars = " .:-=+*#%@"
    hi = float(abs(m).max()) or 1.0
    for row in m:
        line = "".join(chars[min(9, int(abs(float(v)) / hi * 9.999))] for v in row)
        print("      |" + line + "|")


def grid_rows(m, nd=0):
    """把矩阵渲染成等宽的行字符串，方便并排显示。nd = 保留几位小数。"""
    w = 8 if nd == 0 else 9
    fmt = "{:>" + str(w) + "." + str(nd) + "f}"
    rows = []
    for row in m:
        cells = []
        for v in row:
            v = float(v)
            if v == 0:
                v = 0.0          # 把 -0.0 归成 0.0，否则会打出难看的 "-0"
            cells.append(fmt.format(v))
        rows.append("".join(cells))
    return rows


def print_grid(m, label, nd=0):
    print(f"      {label}")
    for line in grid_rows(m, nd):
        print("    " + line)


def print_conv_step(patch, kernel, products, result, where, nd=0):
    """把一次卷积的中间过程完整打出来：窗口、核、逐项乘积、求和。"""
    p = grid_rows(patch, nd)
    k = grid_rows(kernel, nd)
    q = grid_rows(products, nd)

    print(f"    · 把核盖在 {where}：")
    print()
    print("          窗口盖住的 9 个像素           核的 9 个数              逐项相乘")
    for i in range(3):
        if i == 1:
            print("      " + p[i] + "   × " + k[i] + "   = " + q[i])
        else:
            print("      " + p[i] + "     " + k[i] + "     " + q[i])
    print()
    print("      9 个乘积加起来：")
    terms = []
    for v in products.flatten():
        v = float(v)
        if v == 0:
            v = 0.0
        terms.append(f"({v:.{nd}f})" if v < 0 else f"{v:.{nd}f}")
    print("        " + " + ".join(terms))
    print(f"        = {result:.{nd}f}")
    print()


# ---------------------------------------------------------------- 1
def part1():
    """一张图在程序里到底是什么。"""
    title(1, "一张图在程序里到底是什么")

    raw = Flowers102(str(cfg.DATA_DIR), split="train", download=False)
    img, _ = raw[0]
    w, h = img.size

    print(f"  磁盘上的一张图：{w} x {h} 像素，颜色模式 {img.mode}")
    print("  它在内存里其实是三个格子阵：红一个、绿一个、蓝一个（这就是 3 个【通道】）")
    print()

    import numpy as np
    arr = np.array(img)
    print(f"  转成数组看形状：{arr.shape}   ←（高, 宽, 通道）")
    print(f"  每个格子的值落在 [{arr.min()}, {arr.max()}] 之间")
    print("  这是一张声明为 0~255 的整数图（这张碰巧最亮只到 "
          f"{arr.max()}）—— 这就是【像素值】。")
    print()

    print("  但 PyTorch 要的形状是（通道, 高, 宽），而且值要缩到 0~1 上下。")
    print("  这由 ToTensor() 完成。再看缩图那一步的影响：")

    tf_big = transforms.Compose([transforms.ToTensor()])
    tf_small = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    t_big, t_small = tf_big(img), tf_small(img)

    print()
    print(f"  不缩图   ：{tuple(t_big.shape)}   元素个数 {t_big.numel():>9,}")
    print(f"  缩到 {cfg.IMG_SIZE} ：{tuple(t_small.shape)}   元素个数 {t_small.numel():>9,}")
    print(f"  要算的数据少了 {t_big.numel() / t_small.numel():.0f} 倍。")
    print(f"  缩图 = 用精度换速度。CPU 上这一步没法省。")


# ---------------------------------------------------------------- 2
def part2():
    """卷积核：一次算给你看。"""
    import numpy as np

    title(2, "卷积核：一次算给你看")

    print("  上一次我一直在给你看【图】，一次数都没算过 —— 那是我没讲好。")
    print("  这次把镜头拉到最里面：从 9 个数字开始，一步一步算出来。")
    print()

    edge = np.array([[-1., 0., 1.],
                     [-1., 0., 1.],
                     [-1., 0., 1.]])
    blur = np.ones((3, 3)) / 9.0

    print("  先看「卷积核」到底是个什么东西。它就是一个 3×3 的小表格：")
    print()
    print_grid(edge, "一个卷积核（我给它起名叫 edge）")
    print()
    print("  就 9 个数字，没别的。这 9 个数叫【权重】，意思是「各占多重」。")
    print("  训练网络的时候，学的就是这个表格里的 9 个数。")
    print("  人只是决定了「用 3×3」，具体填什么数是网络自己找出来的。")
    print()

    # ------------------------------ 手算第一个格子
    print("=" * 66)
    print("  第一步：手算一个输出格子")
    print("=" * 66)

    img = np.array([[1., 2., 0., 5., 5.],
                    [0., 3., 0., 5., 5.],
                    [2., 1., 0., 5., 5.],
                    [0., 0., 0., 5., 5.],
                    [1., 1., 0., 5., 5.]])

    print("  为了能算得动，先拿一张 5×5 的小图（真图是 96×96，机制完全一样）：")
    print()
    print_grid(img, "一张 5×5 的小图")
    print()

    patch = img[0:3, 0:3]
    products = patch * edge
    print_conv_step(patch, edge, products, products.sum(), "左上角 (0,0)")

    out00 = products.sum()
    print(f"  所以输出的左上角那一格，就是 {out00:.0f}。")
    print("  顺便留意核的中间那一列：全是 0。填 0 等于说「中间这列像素我不关心」，")
    print("  所以那 3 个像素乘完是 0，对结果一点贡献都没有。")
    print("  核里有几个 0，就等于「有几个位置我不看」。")
    print()

    # ------------------------------ 滑动
    print("=" * 66)
    print("  第二步：把窗口往右挪一格，再算一次")
    print("=" * 66)

    patch2 = img[0:3, 1:4]
    products2 = patch2 * edge
    print_conv_step(patch2, edge, products2, products2.sum(), "左上角右移一格，也就是 (0,1)")

    out01 = products2.sum()
    print(f"  这次得到 {out01:.0f}。")
    print(f"  仅仅往右挪了一格，结果就从 {out00:.0f} 变成了 {out01:.0f}。")
    print("  为什么会差这么多？因为新窗口里进来了第三列那排 5——")
    print("  而核最右边那一列是 +1，它专门给右侧的像素记正分。")
    print()
    print("  这里有个容易忽略的事实：两次用的 9 个像素里，有 6 个是同一个像素")
    print("  （窗口往右挪一格，只有最左边一列被换掉，右边两列是重叠的）。")
    print("  所以滑动不是重新取样，是【复用】——")
    print("  这也正是卷积比全连接省参数的原因。")
    print()

    # ------------------------------ 扫完
    print("=" * 66)
    print("  第三步：滑到底，得到整张特征图")
    print("=" * 66)

    out = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            out[r, c] = (img[r:r + 3, c:c + 3] * edge).sum()

    print("  窗口从左上角出发，一行一行滑过去。")
    print(f"  5×5 的图，窗口 3×3，每行能放 3 个位置、共 3 行 —— 所以输出是 3×3：")
    print()
    print_grid(out, "整张特征图（边缘核的输出）")
    print()

    edge_t = torch.tensor(edge, dtype=torch.float32).view(1, 1, 3, 3)
    with torch.no_grad():
        ref = F.conv2d(
            torch.tensor(img, dtype=torch.float32).view(1, 1, 5, 5), edge_t
        )[0, 0].numpy()
    print(f"  拿 PyTorch 自己算一遍对不对得上：{np.allclose(out, ref)}")
    print("  对上了。说明我手算的这套流程，就是 PyTorch 内部在做的事。")
    print()

    print("  顺便回答一个你迟早会问的问题：为什么输出比输入小了 2 格？")
    print("  因为窗口有 3 格宽，它没法盖住「只往外伸出去一格」的位置。")
    print(f"  5 - 3 + 1 = 3，这就是 {tuple(img.shape)} 变成 {tuple(out.shape)} 的原因。")
    print("  想让它不变小，就在图的四周【补一圈 0】（padding=1），")
    print("  补完之后 5+2=7，7-3+1=5，输出又回到 5×5。")
    print()

    # ------------------------------ 换核
    print("=" * 66)
    print("  第四步：同一个位置，换一个核")
    print("=" * 66)

    print("  换成一个「取平均」的核（9 个数都是 1/9）：")
    print()
    print_grid(blur, "平均核 blur", nd=3)
    print()
    products3 = patch * blur
    print_conv_step(patch, blur, products3, products3.sum(), "还是左上角 (0,0)", nd=3)

    print(f"  同一个位置、同一批像素，这次得到 {products3.sum():.3f}。")
    print("  它算的是「这 9 个像素平均有多亮」，跟边缘一点关系都没有。")
    print()
    print("  这就是卷积的核心一句话：")
    print("  【核里的数字决定它关心什么；滑动只负责问遍每一个位置。】")
    print("  换核 = 换一双眼睛。上面这 9 个数，就是那双眼睛的全部。")
    print()

    # ------------------------------ 落在真图上
    print("=" * 66)
    print("  第五步：拿真实的花卉照片走一遍")
    print("=" * 66)

    raw = Flowers102(str(cfg.DATA_DIR), split="train", download=False)
    real = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
    ])(raw[0][0])[0]                      # 只取红色通道，省得同时看三张

    S = cfg.IMG_SIZE
    edge_t = torch.tensor(edge, dtype=torch.float32).view(1, 1, 3, 3)
    with torch.no_grad():
        full = F.conv2d(real.view(1, 1, S, S), edge_t, padding=1)[0, 0]

    # 挑一个响应最剧烈的位置来手算。随机挑很容易挑到一片纯色区域，
    # 那样算出来的数接近 0，演示就没说服力了。
    inner = full[1:-1, 1:-1].abs()
    flat = int(inner.argmax())
    r0 = flat // inner.shape[1]
    c0 = flat % inner.shape[1]

    rp = real[r0:r0 + 3, c0:c0 + 3]
    rprod = rp * edge_t[0, 0]
    print(f"  先在整张图上扫一遍，找出【响应最剧烈】的那个位置——")
    print(f"  它落在 (行 {r0}, 列 {c0})，也就是花的轮廓上。就从这里剪一个 3×3 窗口。")
    print("  这张图已经被 ToTensor 缩到 0~1 之间了，所以数字都是小数：")
    print()
    print_conv_step(rp.numpy(), edge, rprod.numpy(), rprod.sum().item(),
                    f"(行 {r0}, 列 {c0})", nd=3)

    hand = rprod.sum().item()
    torch_out = full[r0 + 1, c0 + 1].item()     # padding=1 时，窗口左上角在 (r0,c0) 对应输出 (r0+1,c0+1)
    print(f"  手算：{hand:+.6f}")
    print(f"  PyTorch：{torch_out:+.6f}")
    print(f"  两个数一模一样：{abs(hand - torch_out) < 1e-6}")
    print("  真实图片上同样成立 —— 说到底它就只有「乘一乘、加一加」这一件事。")
    print()
    print(f"  顺便：如果你刚才自己挑了个纯色的位置，结果会接近 0。")
    print(f"  这不是算错了，那正是边缘核在说「这一片什么变化都没有」。")
    print()

    # ------------------------------ 看图案
    print("=" * 66)
    print("  最后：把边缘核扫过整张花，看看能不能看出轮廓")
    print("=" * 66)

    small = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])(raw[0][0])[0]
    with torch.no_grad():
        e_map = F.conv2d(
            small.view(1, 1, 32, 32), edge_t, padding=1,
        )[0, 0]

    print("  先把原图（32×32，只取红色通道）看一眼：")
    show(small)
    print()
    print("  再用刚才那个边缘核扫一遍，得到的东西长这样：")
    show_signed(e_map)
    print()
    print(f"  这张图里最亮和最暗的格子分别是 {e_map.max():+.3f} 和 {e_map.min():+.3f}，")
    print("  接近 0 的位置画成了空格。你会看到亮暗都挤在花朵的轮廓线上——")
    print("  因为核只看「左右有没有变化」，而轮廓正是变化最剧烈的地方。")
    print()
    print("  这就是卷积网络第一层干的事。它不认花瓣也不认颜色，")
    print("  它只把「哪里变化剧烈」标出来。深层的层拿这些轮廓再组合，")
    print("  才慢慢拼出花瓣、花蕊这种复杂东西。")


# ---------------------------------------------------------------- 3
def part3():
    """池化与通道：形状是怎么一路变小的。"""
    title(3, "池化与通道：形状是怎么一路变小的")

    model = SmallCNN(num_classes=20)
    model.eval()

    records = []

    def hook(mod, inp, out):
        name = mod.__class__.__name__
        if name in ("Conv2d", "MaxPool2d", "AdaptiveAvgPool2d"):
            records.append((name, tuple(out.shape)))

    handles = [m.register_forward_hook(hook) for m in model.modules() if m is not model]
    with torch.no_grad():
        model(torch.zeros(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE))
    for h in handles:
        h.remove()

    print(f"  喂进去的是 (1, 3, {cfg.IMG_SIZE}, {cfg.IMG_SIZE})"
          f" —— 1 张图，3 通道，{cfg.IMG_SIZE}×{cfg.IMG_SIZE} 像素")
    print()
    print(f"  {'层':<22}{'输出形状':<24}{'空间尺寸':>8}")
    print("  " + "-" * 54)
    for name, shape in records:
        _, c, h, w = shape
        print(f"  {name:<22}{str(shape):<24}{h}x{w:>5}")

    print()
    print("  盯住两列看：")
    print(f"    · 空间尺寸 {cfg.IMG_SIZE} → 48 → 24 → 12：每过一个 block 减半，靠 MaxPool。")
    print("    · 通道数 3 → 32 → 64 → 128：每过一个 block 翻倍。")
    print("  一边变矮变窄，一边变厚。厚出来的那一叠，就是越来越抽象的特征。")
    print()
    print("  为什么不一直保持原尺寸？因为越往后的层，要看的范围越大——")
    print("  12x12 上的一个点，对应回原图是 8x8 的一片。缩过之后，")
    print("  同一个 3x3 的窗口才能看住一整片花瓣，而不是一个像素。")
    print()
    print("  最后的 AdaptiveAvgPool2d 把 128x12x12 压成 128x1x1——")
    print("  不管特征图多大，每层都总结成一个数，省掉一坨全连接参数。")


# ---------------------------------------------------------------- 4
def part4():
    """迁移学习为什么能成立。"""
    title(4, "迁移学习为什么能成立（本项目的心脏）")

    print("  先问一个问题：ImageNet 里根本没有 Flowers102 这 102 种花，")
    print("  那借来的网络凭什么认得花？")
    print()
    print("  做个实验回答它：把预训练的主干只当【特征提取器】用，")
    print("  三张图各变成一个向量，再算它们之间的余弦相似度。")
    print("  越接近 1 表示夹角越小，也就是网络眼里这两张图越像。")
    print()

    raw = Flowers102(str(cfg.DATA_DIR), split="test", download=False)
    labels = labels_of(raw)

    first_cls = labels[0]
    idx_a = labels.index(first_cls)                                  # 第一类的第一张
    idx_b = labels.index(first_cls, idx_a + 1)                        # 第一类的第二张
    other_cls = next(y for y in labels if y != first_cls)
    idx_c = labels.index(other_cls)                                   # 另一类的第一张

    tf = build_transforms(cfg.IMG_SIZE, train=False)

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(*list(backbone.children())[:-1])   # 去掉最后那层分类
    extractor.eval()

    with torch.no_grad():
        feats = torch.cat(
            [extractor(tf(raw[i][0]).unsqueeze(0)).flatten(1) for i in (idx_a, idx_b, idx_c)],
            dim=0,
        )
    sim = F.cosine_similarity(feats[0].unsqueeze(0), feats, dim=1)

    print(f"  样本 A、B：同一类（原始标签都是 {first_cls}）")
    print(f"  样本 C   ：另一类（原始标签 {other_cls}）")
    print()
    print(f"  相似度  A vs B（同类）：{sim[1].item():+.4f}")
    print(f"  相似度  A vs C（异类）：{sim[2].item():+.4f}")
    print(f"  两者的差        ：{(sim[1] - sim[2]).item():+.4f}")
    print()
    print("  这个网络从没见过今天这 20 种花，但它给出的特征向量，")
    print("  已经把「同一类」和「不同类」拉开了。")
    print()
    print("  原因：它在 ImageNet 上被迫学会的那些东西（边缘、纹理、")
    print("  颜色搭配、基本形状）是所有图像共通的，不专属某一类图。")
    print("  所以我们只要把它最后一层换掉，教它把这套通用特征")
    print("  映射到我们这 20 个类别上就行。前面 11.19 M 参数原封不动。")
    print()
    print("  这就是【迁移学习】。真实工作中 90% 的图像项目是这么起步的。")


# ---------------------------------------------------------------- 5
def part5():
    """数据增强到底改了什么。"""
    title(5, "数据增强到底改了什么")

    raw = Flowers102(str(cfg.DATA_DIR), split="train", download=False)
    img, _ = raw[0]

    tf_train = build_transforms(cfg.IMG_SIZE, train=True)
    tf_eval = build_transforms(cfg.IMG_SIZE, train=False)

    print("  同一张图，用【训练用】的变换连取 3 次：")
    for k in range(3):
        x = tf_train(img)
        print(f"    第{k + 1}次   像素均值 {x.mean():+.4f}   标准差 {x.std():.4f}")
    print("    → 三次都不一样。因为里面藏着随机翻转、随机旋转。")
    print("      每次取图，网络看到的是同一朵花的另一个角度。")
    print("      等于凭空把数据变多了 —— 每类只有 20 张图时，这是唯一能救命的手段。")
    print()

    print("  同一张图，改用【评估用】的变换连取 3 次：")
    xs = []
    for k in range(3):
        x = tf_eval(img)
        xs.append(x)
        print(f"    第{k + 1}次   像素均值 {x.mean():+.4f}   标准差 {x.std():.4f}")
    same = torch.allclose(xs[0], xs[1]) and torch.allclose(xs[1], xs[2])
    print(f"    → 三次完全一致：{same}")
    print("      评估时必须这样。否则同一张图两次评分不同，")
    print("      误差比你要比较的差距还大，实验就白做了。")
    print()
    print("  顺带一个坑：归一化用的均值/标准差是 ImageNet 的固定数字，")
    print("  不能改成这批花自己的统计量。因为预训练的主干只认那套刻度，")
    print("  换了之后，在它眼里你的图就是另一种颜色。")


# ---------------------------------------------------------------- 6
def part6():
    """过拟合在图像任务里长什么样。"""
    title(6, "过拟合在图像任务里长什么样")

    path = cfg.OUT_DIR / "history_scratch.json"
    if not path.exists():
        print("  还没跑过 scratch，先执行：")
        print("      python train.py --mode scratch --epochs 12 --tag scratch")
        return

    history = json.loads(path.read_text(encoding="utf-8"))

    print("  这是台阶 1（从零训练的小 CNN）每一轮的账本：")
    print()
    print(f"  {'轮':>3}  {'训练准确率':>10}  {'验证准确率':>10}  {'缺口':>8}")
    print("  " + "-" * 38)
    for r in history:
        gap = r["train_acc"] - r["val_acc"]
        print(f"  {r['epoch']:>3}  {r['train_acc'] * 100:>9.1f}%  "
              f"{r['val_acc'] * 100:>9.1f}%  {gap * 100:>+7.1f}%")

    first, last = history[0], history[-1]
    g_first = (first["train_acc"] - first["val_acc"]) * 100
    g_last = (last["train_acc"] - last["val_acc"]) * 100
    print()
    print(f"  头一轮缺口 {g_first:+.1f}%，末一轮缺口 {g_last:+.1f}%。")
    print()
    print("  这就是【过拟合】在图像任务里的长相：")
    print(f"  训练准确率一路爬到 {last['train_acc'] * 100:.1f}%，"
          f"验证准确率却在 {last['val_acc'] * 100:.1f}% 附近来回晃。")
    print("  模型不是没学到东西，它是在背训练集 —— 背得越熟，对新图越不管用。")
    print("  两个数字之间的缺口，就是它在背书的程度。")
    print()

    tpath = cfg.OUT_DIR / "history_transfer.json"
    if tpath.exists():
        th = json.loads(tpath.read_text(encoding="utf-8"))
        tl = th[-1]
        print(f"  对照迁移学习版末轮：train {tl['train_acc'] * 100:.1f}% / "
              f"val {tl['val_acc'] * 100:.1f}%，"
              f"缺口只有 {(tl['train_acc'] - tl['val_acc']) * 100:.1f}%。")
        print("  因为它借来的特征本来就不是为这 320 张图量身定做的，想背也背不动，")
        print("  只能去学真正通用的规律。这是迁移学习的第二个好处。")


PARTS = {1: part1, 2: part2, 3: part3, 4: part4, 5: part5, 6: part6}


if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else "all"

    print()
    print("花卉项目 · 概念预备")
    print("这不是作业，也没有要你背的东西。撞到哪个词卡住了，就来看那一节。")

    cfg.seed_everything(cfg.SEED)

    if want == "all":
        for n in sorted(PARTS):
            PARTS[n]()
    elif want.isdigit() and int(want) in PARTS:
        PARTS[int(want)]()
    else:
        print(f"\n  不认识这个编号：{want}。可以填 1~6 或 all。")

    print()
    print("-" * 66)
    print("  看完了？回去跑 T1：python train.py --mode scratch --epochs 3 --tag 试探")
    print("-" * 66)
