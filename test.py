import yaml
import torch
from data.data_loader import get_dataloader
from models.model_builder import build_model
from evaluation.metrics import calculate_acc, get_confusion_matrix

def main():
    with open("./configs/cifar10_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloader(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(f"{cfg['save_dir']}/best_model.pth"))
    model.eval()

    test_pred, test_label = [], []
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            pred = torch.argmax(logits, dim=1)
            test_pred.extend(pred.cpu().numpy())
            test_label.extend(label.cpu().numpy())
    acc = calculate_acc(test_pred, test_label)
    cm = get_confusion_matrix(test_pred, test_label)
    print(f"Final Test Accuracy = {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
