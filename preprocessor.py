import torch
from decouple import config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch
from PIL import Image
import os


class TrainDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.image_folder = ImageFolder(root=data_folder, transform=transform)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        return self.image_folder[idx]




class ValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        images = []
        labels = []
        labels_file = os.path.join(self.root, 'ILSVRC2010_validation_ground_truth.txt')
        with open(labels_file, 'r') as file:
            labels = [int(line.strip())-1 for line in file]

        for filename in os.listdir(self.root): 
            if filename.endswith(".JPEG"):
                image_path = os.path.join(self.root, filename)
                images.append(image_path)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            if self.transform:
                image = self.transform(image)
        return image, label



def load_data_cifar(preprocess, train, test, device):
    # train = CIFAR10(root, download=True, train=True)
    train_indices, validation_indices = train_test_split(range(len(train)), test_size=0.2, random_state=2295)
    train_set = torch.utils.data.Subset(train, train_indices)
    validation_set = torch.utils.data.Subset(train, validation_indices)
    train_set.dataset.transform = preprocess
    validation_set.dataset.transform = preprocess
    # test = CIFAR10(root, download=True, train=False, transform=preprocess)

    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test.classes]).to(device)
    print(f'trainset size {len(train_set)}')
    print(f'validation_set size {len(validation_set)}')
    print(f'test size {len(test)}')
    
    batch_size = int(config('batch_size'))
    trainloaders = [torch.utils.data.DataLoader(train_set, batch_size=int(config('batch_size')), shuffle=True) for i in range(int(config('opt')))]
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return trainloaders, validation_loader, test_loader


def load_data_places(preprocess, train_data, val_data, test_data, device):
    trainloaders = [train_data.pytorch(num_workers = 4, shuffle = True, transform = {'images': preprocess, 'labels': None}, batch_size = int(config('batch_size')), decode_method = {'images': 'pil'})  for i in range(int(config('opt')))]
    validation_loader = val_data.pytorch(num_workers = 4, shuffle = False, transform = {'images': preprocess, 'labels': None}, batch_size = int(config('batch_size')), decode_method = {'images': 'pil'})
    test_loader = test_data.pytorch(num_workers = 4, shuffle = False, transform = {'images': preprocess, 'labels': None}, batch_size = int(config('batch_size')), decode_method = {'images': 'pil'})

    return trainloaders, validation_loader, test_loader



def load_data_imagenet( train_set, val_dataset, test_dataset, device):
    torch.manual_seed(42)

    print(f'trainset size {len(train_set)}')
    print(f'validation_set size {len(val_dataset)}')
    print(f'test size {len(test_dataset)}')
    
    trainloaders = [torch.utils.data.DataLoader(train_set, batch_size=int(config('batch_size')), shuffle=True) for i in range(int(config('opt')))]
    val_loader = DataLoader(val_dataset, batch_size=int(config('batch_size')), shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=int(config('batch_size')), shuffle=False, num_workers=8, pin_memory=True)

    return trainloaders, val_loader, test_loader