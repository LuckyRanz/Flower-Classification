"""训练主循环。

你在第十四课手写过的那套"五步曲"（清零 → 前向 → 算损失 → 反向 → 更新），
就是下面 run_epoch 里的那几行。这里只是把它包装成工程上能直接用的样子：
数据怎么来、什么时候存模型、日志怎么写。

用法：
    python train.py                      # 按 config.py 的设置跑
    python train.py --mode transfer      # 临时换成迁移学习
    python train.py --epochs 3    
           # 先跑 3 轮看看通不通
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import config as cfg
from data_utils import build_loaders
from model import build_model, count_params, lock_frozen_batchnorm


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """跑一遍数据，返回 (平均损失, 准确率)。

    三个容易写错的地方，这里都处理了：
      1. 损失累加时要乘上 batch 里真实的样本数。最后一个 batch 常常不满，
         直接对 batch 的 loss 取平均，会被它带偏。
      2. 评估时必须 torch.no_grad()，否则会白白建一张计算图，又慢又吃内存。
      3. 迁移学习时，model.train() 会把"被冻住的主干"里的 BatchNorm 也一并
         设成训练模式，于是它开始拿花卉图去覆盖 ImageNet 学来的统计量。
         所以要跟着按回 eval —— 参数不动，刻度也不动。
         （细节见 model.py 里 lock_frozen_batchnorm 的说明）
    """
    if train:
        model.train()
        lock_frozen_batchnorm(model)
    else:
        model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    with (torch.enable_grad() if train else torch.no_grad()):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)            # 前向
            loss = criterion(logits, yb)  # 算损失

            if train:
                optimizer.zero_grad()     # 清零
                loss.backward()           # 反向
                optimizer.step()          # 更新

            n = xb.size(0)
            total_loss += loss.item() * n
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_n += n

    return total_loss / total_n, total_correct / total_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["scratch", "transfer"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--classes", type=int, default=None)
    p.add_argument("--tag", default=None,
                   help="给这次实验起个短名字，会写进输出文件名，方便回头对比")
    args = p.parse_args()

    if args.mode:
        cfg.MODE = args.mode
    if args.epochs:
        cfg.EPOCHS = args.epochs
    if args.img_size:
        cfg.IMG_SIZE = args.img_size
    if args.classes:
        cfg.NUM_CLASSES = args.classes

    cfg.seed_everything(cfg.SEED)
    device = torch.device("cpu")     # 这台机器没有独显，明写出来，免得以后忘了

    lr = args.lr or (cfg.LR if cfg.MODE == "scratch" else cfg.LR_FINETUNE)
    tag = args.tag or cfg.MODE

    print("=" * 66)
    print(f"实验配置：{cfg.describe()}")
    print(f"学习率 {lr}")
    print("=" * 66)

    train_loader, val_loader, _ = build_loaders(cfg)
    print(f"数据：训练 {len(train_loader.dataset)} 张 / "
          f"验证 {len(val_loader.dataset)} 张")
    print(f"     每轮 {len(train_loader)} 步（batch_size={cfg.BATCH_SIZE}）")

    model = build_model(cfg.MODE, cfg.NUM_CLASSES).to(device)
    total, trainable = count_params(model)
    print(f"模型：{cfg.MODE}  总参数 {total} M  可训练 {trainable} M")
    if cfg.MODE == "transfer":
        print("      ↑ 可训练参数只占一小撮：主干被冻住了，只有最后那层在学")
        print(f"      ↑ 顺手锁住 {lock_frozen_batchnorm(model)} 层 BatchNorm 的统计量"
              "（见 model.py 的说明）")
    print()

    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    # 学习率随训练慢慢降下来，最后收敛得更稳
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val, best_path = 0.0, cfg.OUT_DIR / f"best_{tag}.pt"

    t_start = time.time()
    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc,
                        "seconds": round(time.time() - t0, 1)})

        flag = ""
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model_state": model.state_dict(),
                        "mode": cfg.MODE, "num_classes": cfg.NUM_CLASSES,
                        "img_size": cfg.IMG_SIZE, "val_acc": va_acc},
                       best_path)
            flag = "  ← 目前最好，已保存"

        print(f"  epoch {epoch:>2}/{cfg.EPOCHS}  "
              f"train {tr_loss:.4f} / {tr_acc * 100:5.1f}%   "
              f"val {va_loss:.4f} / {va_acc * 100:5.1f}%   "
              f"{history[-1]['seconds']:>5.1f}s{flag}")

    total_min = (time.time() - t_start) / 60
    print()
    print(f"训练结束，共 {total_min:.1f} 分钟。最好的验证准确率 {best_val * 100:.1f}%")
    print(f"模型已存到 {best_path}")

    log_path = cfg.OUT_DIR / f"history_{tag}.json"
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"训练日志已存到 {log_path}")
    print()
    print("下一步：python evaluate.py --tag " + tag)


if __name__ == "__main__":
    main()
