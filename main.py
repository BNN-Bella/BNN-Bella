
import os
import open_clip
import random
import ast
import torch
import deeplake
from torchvision.datasets import CIFAR10, CIFAR100
from decouple import config
from torchvision.datasets import CIFAR10
from utils import Paths
from preprocessor import load_data_cifar, load_data_places, load_data_imagenet, ValDataset, TrainDataset
from wilds import get_dataset
from bayes_wrap import generate_freezed_particles, train_model_wrap_cifar, generate_lora_particles, train_model_wrap_places, train_model_wrap_camelyon




''' -----------------------   Set path ------------------------------'''
paths = Paths(config)
paths.create_path()


''' -----------------------   loading CLIP ViT ------------------------------'''
device = f"cuda:{config('device')}" if torch.cuda.is_available() else "cpu"
mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')



if config('dataset_name').upper() == "CAMELYON":
    download_path = os.path.expanduser(f"{config('dataset_path')}/{config('dataset_name').lower()}")
    dataset = get_dataset(dataset="camelyon17", download=True,  root_dir=download_path)
    train_data = dataset.get_subset(
        "train",
        transform=preprocess
    )

    val_data = dataset.get_subset(
        "val",
        transform=preprocess
    )

    test_data = dataset.get_subset(
        "test",
        transform=preprocess
    )
    print('camelyon loaded')
    trainloaders = [torch.utils.data.DataLoader(train_data, batch_size=int(config('batch_size')), shuffle=True) for i in range(int(config('opt')))]
    valloader = torch.utils.data.DataLoader(val_data, batch_size=int(config('batch_size')), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=int(config('batch_size')), shuffle=False) 
elif config('dataset_name').upper() == "CIFAR10":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser(f"{config('dataset_path')}/{config('dataset_name').lower()}")
    train = CIFAR10(root, download=True, train=True)
    test = CIFAR10(root, download=True, train=False, transform=preprocess)

    print('cifar10 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "CIFAR100":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser(f"{config('dataset_path')}/{config('dataset_name').lower()}")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)


    print('cifar100 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "DOMAINNET":

    ''' -----------------------   Loading the Data   ----------------------- '''
    train_data = deeplake.load("hub://activeloop/domainnet-real-train")
    test_data = deeplake.load("hub://activeloop/domainnet-real-test")

    print('Domainnet has been loaded')
    print(f'len train is {len(train_data)}')
    print(f'len test is {len(test_data)}')

    trainloaders, validation_loader, test_loader = load_data_places(preprocess, train_data, test_data, test_data, device)

elif config('dataset_name').upper() == "IMAGENET":
    ''' -----------------------   Loading the Data   ----------------------- '''

    train_set = TrainDataset(data_folder=f"{config('dataset_path')}/{config('dataset_name').lower()}/train", transform=preprocess)
    val_dataset = TrainDataset(data_folder=f"{config('dataset_path')}/{config('dataset_name').lower()}/val", transform=preprocess)    

    test_dataset = TrainDataset(data_folder=f"{config('dataset_path')}/{config('dataset_name').lower()}/test", transform=preprocess)
    trainloaders, validation_loader, test_loader = load_data_imagenet( train_set, val_dataset, test_dataset, device)

    print('ImageNet loaded')


print('-----------------------------------Training has been started-------------------- ')

for i in range(int(config('no_run'))):

    print(f"training model {i} has been started")

    device = f"cuda:{config('device')}" if torch.cuda.is_available() else "cpu"
    mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    particles = generate_freezed_particles(mdl , int(config('opt')), device)
    delta_models = generate_lora_particles(particles)
    
    if config('dataset_name').upper() == "CAMELYON":
        train_model_wrap_camelyon(delta_models, trainloaders, valloader, i, device, config)
        
    elif  config('dataset_name').upper() == "CIFAR10":
        train_model_wrap_cifar(delta_models, trainloaders, validation_loader, i, device, config)
    
    elif  config('dataset_name').upper() == "CIFAR100":
        train_model_wrap_cifar(delta_models, trainloaders, validation_loader, i, device,config)

    elif config('dataset_name').upper() == "DOMAINNET":
        print(f"training {str(config('dataset_name'))} has been started")
        train_model_wrap_places(delta_models, trainloaders, validation_loader, i, config)

    print(f'training model_{i} is completed')


