
import os
import open_clip
import random
import torch
import deeplake
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from decouple import config
from torchvision.datasets import CIFAR10
from model import evaluate_model_cam_ensemble_freeze, evaluate_model_cifar_ensemble_freeze
from utils import Paths
from preprocessor import load_data_cifar, load_data_places, load_data_imagenet, ValDataset, TrainDataset
from wilds import get_dataset
from bayes_wrap import generate_freezed_particles, generate_lora_particles



random.seed(2295)

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

#####################################################################################################

mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

particles = generate_freezed_particles(mdl , int(config('opt')), device)
delta_models = generate_lora_particles(particles)

model_address = [f for f in os.listdir('Model/') if f[-3:]=='.pt']


for i, mdl_addr in enumerate(model_address):
    mdl_addr = f'Model/{mdl_addr}'
    fine_tuned_weights = torch.load(mdl_addr, weights_only=True)
    delta_models[i].load_state_dict(fine_tuned_weights)


if config('dataset_name').upper() == "CIFAR100":
    num_classes=100
    acc, ece, nll = evaluate_model_cifar_ensemble_freeze(delta_models, test_loader, num_classes, device)

elif config('dataset_name').upper() == "CIFAR10":
    num_classes=10
    acc, ece, nll = evaluate_model_cifar_ensemble_freeze(delta_models, test_loader, num_classes, device)

elif config('dataset_name').upper() == "CAMELYON":
    num_classes=2    
    acc, ece, nll = evaluate_model_cam_ensemble_freeze(delta_models, test_loader, num_classes, device)

elif config('dataset_name').upper() == "IMAGENET":
    num_classes=1000   
    acc, ece, nll = evaluate_model_cam_ensemble_freeze(delta_models, test_loader, num_classes, device) 

elif config('dataset_name').upper() == "DOMAINNET":
    num_classes=345   
    acc, ece, nll = evaluate_model_cam_ensemble_freeze(delta_models, test_loader, num_classes, device) 


print(f'acc: {acc}, ece: {ece}, nll: {nll}')



#----------------------------------------------------------------------------------
    
# if config('dataset_name').upper() == "CIFAR100":


#     corrupted_address = [f for f in os.listdir("./Data/") if f[-4:]=='.npy']

#     print(f'corrupted is {corrupted_address}')

#     performance=[]
#     for i, corr in enumerate(corrupted_address):
#         if corr != "labels.npy":
#             print(f'noise is {corr.split(".")[0]}')
#             perf=[]
#             corrupted_testset = np.load("Data/" + corr)
#             lbls = np.load("Data/labels.npy")
#             test.data = corrupted_testset
#             test.targets = lbls
#             test.transform = preprocess
#             print(f'len {corr} is {len(test)}')
#             trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)
#             perf.append(corr.split(".")[0])

#             acc, ece, nll = evaluate_model_cam_ensemble_freeze(delta_models, test_loader, device)
#             print(f'acc: {acc}, ece: {ece}, nll: {nll}')
#             a=[round(acc, 4)*100, round(ece, 5), round(nll, 4)]
#             perf.append(a)

#             performance.append(perf)
                
        
#     performance_path = f"Results/cifar100_sev1.json"
#     with open(performance_path, 'w') as fp:
#         json.dump(performance, fp, indent=2)
#     print(performance)

