from torchvision import datasets, transforms

def load_kmnist():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    return dataset
