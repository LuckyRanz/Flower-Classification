"""评估：不只报一个准确率，还要看它错在哪儿。

准确率只回答"对了多少"，不回答"错得有没有道理"。
这个脚本会多给你三样东西：
    · 每类的准确率 —— 有没有哪几种花被整个放弃
    · 混淆矩阵     —— 哪些类别在被互相认错（第十六课你算过这个）
    · 最常混的类别对 —— 这是找问题方向的线索

用法：
    python evaluate.py --tag scratch
"""

import argparse
from collections import Counter

import numpy as np
import torch

import config as cfg
from data_utils import build_loaders
from model import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """把整个数据集跑一遍，收集真实标签和预测标签。"""
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def confusion_matrix(y_true, y_pred, num_classes) -> np.ndarray:
    """自己算一遍混淆矩阵，不用 sklearn——你第十六课手算过，这里就是代码版。

    行 = 真实类别，列 = 预测类别。cm[i, j] 表示"真身是 i、被认成 j"的张数。
    对角线上的数字就是认对的。
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_cm_figure(cm, tag, path, acc: float):
    """把混淆矩阵画成 PNG —— 终端里 20×20 的数字挤成一团，图才看得清。

    这台机器没装 matplotlib，所以用 PIL 手绘（跟课程里那些配图同一套做法）：
    白底、超采样后缩小抗锯齿、微软雅黑。
    行 = 真实类别，列 = 模型猜的类别；对角线上的是猜对的。
    """
    from PIL import Image, ImageDraw, ImageFont

    S = 3                      # 超采样倍数，缩回原尺寸时自带抗锯齿
    _fc = {}

    def font(sz, bold=False):
        k = (sz, bold)
        if k not in _fc:
            path_ttf = ("C:/Windows/Fonts/msyhbd.ttc" if bold
                        else "C:/Windows/Fonts/msyh.ttc")
            _fc[k] = ImageFont.truetype(path_ttf, max(8, sz * S))
        return _fc[k]

    n = cm.shape[0]
    W = 680
    X0, Y0 = 76, 78
    CELL = max(6, min(28, int((W - X0 - 44) / n)))
    show_num = CELL >= 20
    GW = n * CELL
    H = Y0 + GW + 26 + 78

    im = Image.new("RGB", (W * S, H * S), "#FFFFFF")
    d = ImageDraw.Draw(im)

    def P(v):
        return v * S

    def txt(x, y, s, sz=12, color="#1F2328", bold=False, anchor="mm"):
        d.text((P(x), P(y)), s, font=font(sz, bold), fill=color, anchor=anchor)

    def rect(x, y, w, h, fill=None, outline=None, lw=1.2):
        d.rectangle([P(x), P(y), P(x + w), P(y + h)], fill=fill, outline=outline,
                    width=max(1, int(round(lw * S))))

    total = int(cm.sum())
    hit = int(cm.diagonal().sum())
    vmax = max(int(cm.max()), 1)

    txt(40, 30, f"混淆矩阵 · {tag}", sz=16.5, bold=True, anchor="lm")
    txt(40, 54, f"测试集 {total} 张，准确率 {acc * 100:.1f}%"
                f"（对角线 {hit} 张 ÷ {total}）", sz=12, color="#6E7781", anchor="lm")
    txt(40, 72, "行 = 真实类别 ↓      列 = 模型猜的类别 →", sz=11.5,
        color="#98A2AE", anchor="lm")

    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            t = (v / vmax) ** 0.55 if v else 0.0
            if v == 0:
                fill = "#F6F8FA"
            else:
                fill = "#%02X%02X%02X" % tuple(
                    int(255 + (b - 255) * t) for b in (9, 105, 218))
            x, y = X0 + j * CELL, Y0 + i * CELL
            rect(x, y, CELL, CELL, fill=fill, outline="#FFFFFF")
            if v and show_num:
                txt(x + CELL / 2, y + CELL / 2, str(v), sz=12,
                    color="#FFFFFF" if t > 0.45 else "#1F2328")
        rect(X0 + i * CELL, Y0 + i * CELL, CELL, CELL,
             outline="#8250DF", lw=2.0)     # 对角线 = 猜对，单独描边

    for k in range(n):
        if n <= 30 or k % 5 == 0:
            txt(X0 - 10, Y0 + k * CELL + CELL / 2, str(k), sz=11,
                color="#6E7781", anchor="rm")
            txt(X0 + k * CELL + CELL / 2, Y0 + GW + 15, str(k), sz=11,
                color="#6E7781", anchor="mm")

    y1 = Y0 + GW + 44
    txt(40, y1, f"紫框里的 {hit} 格是猜对的；框外的每个格都是一次认错。",
        sz=12, color="#1F2328", anchor="lm")
    txt(40, y1 + 22, "颜色越深张数越多。最值得盯的是框外的深格子 —— 模型老把这两类搞混。",
        sz=11.5, color="#6E7781", anchor="lm")

    im.resize((W, H), Image.LANCZOS).save(path, "PNG", optimize=True)
    return y1 + 22


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="scratch")
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt_path = cfg.OUT_DIR / f"best_{args.tag}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"找不到 {ckpt_path}，先把 train.py 跑完再来")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg.MODE = ckpt["mode"]
    cfg.NUM_CLASSES = ckpt["num_classes"]
    cfg.IMG_SIZE = ckpt["img_size"]

    model = build_model(ckpt["mode"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, _, test_loader = build_loaders(cfg)
    n_test = len(test_loader.dataset)

    y_true, y_pred = collect_predictions(model, test_loader, device)
    acc = (y_true == y_pred).mean()
    baseline = 1.0 / cfg.NUM_CLASSES

    print("=" * 66)
    print(f"评估 [{args.tag}]  模式 {ckpt['mode']}  图像 {cfg.IMG_SIZE}px  "
          f"{cfg.NUM_CLASSES} 类")
    print("=" * 66)
    print(f"测试集 {n_test} 张")
    print(f"  准确率        {acc * 100:5.1f}%")
    print(f"  瞎猜基线      {baseline * 100:5.1f}%   （{cfg.NUM_CLASSES} 类均匀乱猜）")
    print(f"  相对基线      {acc / baseline:5.1f} 倍")
    if "val_acc" in ckpt:
        print(f"  训练时的验证准确率 {ckpt['val_acc'] * 100:.1f}%"
              f"   （和测试集差 {(ckpt['val_acc'] - acc) * 100:+.1f} 个百分点）")

    # ---- 每类准确率 ----
    cm = confusion_matrix(y_true, y_pred, cfg.NUM_CLASSES)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    order = np.argsort(per_class)

    print()
    print("每类准确率——最差的 5 个：")
    for i in order[:5]:
        print(f"  类别 {i:>3}：{per_class[i] * 100:5.1f}%   （{cm[i].sum()} 张里对了 {cm[i, i]} 张）")
    print("每类准确率——最好的 5 个：")
    for i in order[::-1][:5]:
        print(f"  类别 {i:>3}：{per_class[i] * 100:5.1f}%   （{cm[i].sum()} 张里对了 {cm[i, i]} 张）")

    # ---- 最常混的类别对 ----
    pairs = []
    for i in range(cfg.NUM_CLASSES):
        for j in range(cfg.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], i, j))
    pairs.sort(reverse=True)

    print()
    print("最容易认错的几对（真身 → 被认成）：")
    for cnt, i, j in pairs[:8]:
        print(f"  类别 {i:>3} → {j:>3}  错了 {cnt:>3} 张"
              f"   （该类的 {cnt / max(cm[i].sum(), 1) * 100:.0f}% 都跑这儿来了）")

    print()
    print("矩阵对角线占比：%.1f%% —— 剩下的都散在对角线外面。" % (cm.diagonal().sum() / cm.sum() * 100))
    print()
    print("接下来值得做的：")
    print("  · 把最容易认错的那几对，各挑几张图打开看看——它们到底像在哪")
    print("  · 想想是'图太像'还是'模型没看对地方'，两者的改法完全不同")

    cm_npy = cfg.OUT_DIR / f"cm_{args.tag}.npy"
    cm_png = cfg.OUT_DIR / f"cm_{args.tag}.png"
    np.save(cm_npy, cm)
    save_cm_figure(cm, args.tag, cm_png, acc)

    print()
    print(f"混淆矩阵已存到 {cm_npy}")
    print(f"混淆矩阵图已存到 {cm_png}")
    print("                     ↑ 这个才是给你看的：在 VSCode 左侧点一下就能预览")


if __name__ == "__main__":
    main()
