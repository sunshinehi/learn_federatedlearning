import torchvision

mnist_train = torchvision.datasets.MNIST("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                         download=True)
mnist_test = torchvision.datasets.MNIST("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

cifar_train = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
cifar_test = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)
