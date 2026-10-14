import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def get_cifar10_transform(train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    return transform

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):

        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label
