import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_user, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_user[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_user[i])
    return dict_user


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_user, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_user[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_user[i])
    return dict_user


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 10000
    d = mnist_iid(dataset_train, num)
    print(len(dataset_train))
    print(len(d))
    print(d.keys())
    print(d[1])
