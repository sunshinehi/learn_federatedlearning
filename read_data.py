import torchvision.datasets
from torch.utils.data import DataLoader, Dataset

data_train = torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        print(idxs)
        image, label = self.dataset[self.idxs[item]]
        print(idxs[item])
        return image, label


idxs = [1, 3, 2]
ldr_train = DatasetSplit(data_train, idxs)
l1 = ldr_train[0]
l1 = ldr_train[1]
l1 = ldr_train[2]


