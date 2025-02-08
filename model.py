import torch
import math
import copy
import open_clip
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from decouple import config
from utils import add_noise_to_parameters
import numpy as np
import os
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
import torch.optim.lr_scheduler as lr_scheduler
from utils import cosine_lr, calculate_metrics

from opendelta.utils.inspect import inspect_module_statistics
from bigmodelvis import Visualization

"""
from opendelta import Visualization
from opendelta import LowRankAdapterModel, AdapterModel
from opendelta import LoraModel # use lora as an example, others are same
"""

device = f"cuda:{config('device')}" if torch.cuda.is_available() else "cpu"


def evaluate_model_freeze(model, test_loader, device):
    

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            model = model.cuda()
            img , text = images.to(device), labels.to(device)
            logits = model.backbone_model(img)


            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
    return  all_scores, all_labels    


def evaluate_model_cam_ensemble_freeze(ensemble, test_loader, num_classes, device):


    all_scores = []
    logits_test, targets_test = [], []
    i = 0
    total_nll = 0.0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            
            img , text = images.to(device), labels.to(device)
 
            logits = [] 
            softs, entropies = [],[]
            for model in ensemble:
                model.eval()
                model = model.to(device)
                l = model.backbone_model(img)
                sft = torch.softmax(l, 1)

                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
               
            print(f'\r {i}', end='')
            i +=1


            logits = torch.stack(logits).mean(0)
            probs = torch.softmax(logits, dim=-1)
            logits_test.append(logits.cpu().detach().numpy())
            targets_test.append(text.cpu().detach().numpy())


            # Calculate NLL using probabilities
            loss_func = torch.nn.NLLLoss(reduction="mean")
            nll = loss_func(torch.log(probs), text)  # Calculate NLL for this batch
            total_nll += nll.item() * text.size(0)  


        logits_test = np.concatenate(logits_test, axis=0)
        targets_test = np.concatenate(targets_test, axis=0)

        lgits = torch.tensor(logits_test, dtype=torch.float32)
        targets = torch.tensor(targets_test, dtype=torch.long)
        preds = torch.argmax(lgits, dim=1)
        correct = (preds == targets).sum().item()
        ac = correct / targets.size(0)

        num_classes = 10

        ECE, MCE = calculate_metrics(logits_test, targets_test, num_classes, n_bins=15)
        print(
            '[Calibration - Default T=1] ACC = %.4f, ECE = %.4f' %
            (ac, ECE)
        )   

        average_nll = total_nll / targets.size(0)
        print(f'Average NLL: {average_nll:.4f}')

    return  ac, ECE, average_nll


def evaluate_model_cifar_ensemble_freeze(ensemble, test_loader, num_classes, device):


    all_scores = []
    logits_test, targets_test = [], []
    i = 0
    total_nll = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            
            img , text = images.to(device), labels.to(device)
 
            logits = [] 
            softs, entropies = [],[]
            for model in ensemble:
                model.eval()
                model = model.to(device)
                l = model.backbone_model(img)
                sft = torch.softmax(l, 1)

                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
               
            print(f'\r {i}', end='')
            i +=1


            logits = torch.stack(logits).mean(0)
            probs = torch.softmax(logits, dim=-1)
            logits_test.append(logits.cpu().detach().numpy())
            targets_test.append(text.cpu().detach().numpy())


            # Calculate NLL using probabilities
            loss_func = torch.nn.NLLLoss(reduction="mean")
            nll = loss_func(torch.log(probs), text)  # Calculate NLL for this batch
            total_nll += nll.item() * text.size(0)  


        logits_test = np.concatenate(logits_test, axis=0)
        targets_test = np.concatenate(targets_test, axis=0)

        lgits = torch.tensor(logits_test, dtype=torch.float32)
        targets = torch.tensor(targets_test, dtype=torch.long)
        preds = torch.argmax(lgits, dim=1)
        correct = (preds == targets).sum().item()
        ac = correct / targets.size(0)

        num_classes = 10

        ECE, MCE = calculate_metrics(logits_test, targets_test, num_classes, n_bins=15)
        print(
            '[Calibration - Default T=1] ACC = %.4f, ECE = %.4f' %
            (ac, ECE)
        )   

        average_nll = total_nll / targets.size(0)
        print(f'Average NLL: {average_nll:.4f}')

    return  ac, ECE, average_nll

def evaluate_model_lora_uncertainty(ensemble, test_loader, device):
    """Evalate for lora model 
    to get uncertainty
    """

    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_H, all_E_entropies, all_softs, all_stds = [], [], [], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:

            img , text = images.to(device), labels.to(device)

            logits = [] 
            softs, entropies = [],[]
            for model in ensemble:

                model = model.cuda()
                l = model.backbone_model(img)
                sft = torch.softmax(l, 1)

                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)

            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            # this is stack of particle entropy
            entropies = torch.stack(entropies)
            # this is expected entropies
            E_entropies = entropies.mean(0)
            # get the entropy of expected probabilities
            H = (-softs * torch.log(softs + 1e-8)).sum(1)


            predicted = torch.argmax(softs, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_H.extend(H.cpu().numpy())
            all_E_entropies.extend(E_entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

            print(f'\r {i}', end='')
            i +=1

        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_H = np.array(all_H).tolist()
        all_E_entropies = np.array(all_E_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_H_test":  all_H,
                     "all_E_entropies_test":  all_E_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds
                                                }

    labels_info_path = "Results/Bella_MI_{}.json".format(config('dataset_name'))
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 



def averaging_model(model_address):

    ensemble=[]
    for i in range(len(model_address)):
        mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        # mdl_addr = f'nmdl/mdl_{i}.pt'
        classification_head = get_classification_head()
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)
        net = ImageClassifier(image_encoder, classification_head)
        net.freeze_head()

    
        model_new = copy.deepcopy(net)
        fine_tuned_weights = torch.load("./nmdl/"+ model_address[i])
        model_new.load_state_dict(fine_tuned_weights)
        ensemble.append(model_new)
        print(f'model {i} is loaded from {model_address[i]}')


    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    classification_head = get_classification_head()
    image_encoder = ImageEncoder(model)#, keep_lang=False)
    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()


    average_model = copy.deepcopy(net)

    state_dicts = [mdel.state_dict() for mdel in ensemble]

    average_state_dict = {}
    num_models = len(ensemble)

    # coefficients = [0.1, 0.3, 0.15, 0.25, 0.2]


    for key in ensemble[0].state_dict():
        average_state_dict[key] =sum(state_dict[key] for state_dict in state_dicts) / num_models
        # average_state_dict[key] = sum([coeff * state_dict[key] for coeff, state_dict in zip(coefficients, state_dicts)])#/ len(coefficients)

    average_model.load_state_dict(average_state_dict)

    print('The averaged model will be used for comparison')
    print("")   

    return average_model, ensemble










