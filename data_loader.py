from torch.utils.data import DataLoader
from datasets.cifar10_dataset import CIFAR10Dataset, get_cifar10_transform

def get_dataloader(cfg):
    train_transform = get_cifar10_transform(train=True)
    test_transform = get_cifar10_transform(train=False)

    train_set = CIFAR10Dataset(root=cfg["data_root"], train=True, transform=train_transform)
    test_set = CIFAR10Dataset(root=cfg["data_root"], train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    return train_loader, test_loader

