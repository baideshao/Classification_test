import yaml
import torch
import tqdm
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from data.data_loader import get_dataloader
from models.model_builder import build_model
from losses.custom_loss import build_loss
from evaluation.metrics import calculate_acc
from tools.utils import create_dir


def main():
    # 读取配置
    with open("./configs/cifar10_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    create_dir(cfg["save_dir"])
    os.makedirs("./docs", exist_ok=True)
    os.makedirs("./experiments", exist_ok=True)

    # ============ 日志模块：保存终端输出到 docs/train_log.txt ============
    log_path = "./docs/train_log.txt"
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    sys.stdout = Logger(log_path)
    print(f"===== CIFAR10 训练日志，开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # 用来保存每一轮指标，绘图用
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    train_loader, test_loader = get_dataloader(cfg)
    model = build_model(cfg).to(device)
    criterion = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_acc = 0.0

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0
        train_pred, train_label = [], []
        pbar = tqdm.tqdm(train_loader)
        for img, label in pbar:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            train_pred.extend(pred.detach().cpu().numpy())
            train_label.extend(label.detach().cpu().numpy())
            pbar.set_description(f"Epoch {epoch+1}, Loss:{loss.item():.4f}")
        train_acc = calculate_acc(train_pred, train_label)
        avg_train_loss = train_loss / len(train_loader)

        # 存入列表
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(train_acc)

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f}, Train Loss: {avg_train_loss:.4f}")

        # 每轮测试
        model.eval()
        test_pred, test_label = [], []
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                pred = torch.argmax(logits, dim=1)
                test_pred.extend(pred.cpu().numpy())
                test_label.extend(label.cpu().numpy())
        test_acc = calculate_acc(test_pred, test_label)
        test_acc_list.append(test_acc)
        print(f"Epoch {epoch+1} | Test Acc: {test_acc:.4f}")

        # ✅ 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "./experiments/best_model.pth")
            print(f"✅ 更新最优模型, Best Test Acc: {best_acc:.4f}")


    # ============ 训练结束，绘制Loss/Acc曲线并保存图片 ============
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss Curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(test_acc_list, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./docs/loss_acc_curve.png", dpi=200)
    plt.close()
    print("\n✅ 训练曲线图保存到 docs/loss_acc_curve.png")
    print(f"🏁 训练全部完成，最优测试集准确率 = {best_acc:.4f}")
    print(f"===== CIFAR10 训练日志，结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    sys.stdout.log.close()


if __name__ == "__main__":
    main()
