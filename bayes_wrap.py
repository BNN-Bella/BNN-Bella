

"""Train CIFAR10 with PyTorch."""
import copy
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import grad
from decouple import config
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
from utils import cosine_lr
from bigmodelvis import Visualization
from opendelta import LoraModel
from opendelta.utils.inspect import inspect_module_statistics
import importlib
import re

seed = 113
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



def update_gradiants(all_pgs, h_kernel):

    if np.random.rand() < 0.95:
        return

    if h_kernel is None or h_kernel <= 0:
        h_kernel = 0.001  # 1
    dists = []
    alpha = 0.3  # if t < 100 else 0.0
    new_parameters = [None] * len(all_pgs)

    for i in range(len(all_pgs)):
        new_parameters[i] = {}
        for l, p in enumerate(all_pgs[i].parameters()):
            if p.grad is None:
                new_parameters[i][l] = None
            else:
                new_parameters[i][l] = p.grad.data.new(
                    p.grad.data.size()).zero_()
        for j in range(len(all_pgs)):
            # if i == j:
            #     continue
            for l, params in enumerate(
                    zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                p, p2 = params
                if p.grad is None or p2.grad is None:
                    continue
                if p is p2:
                    dists.append(0)
                    new_parameters[i][l] = new_parameters[i][l] + \
                        p.grad.data
                else:
                    d = (p.data - p2.data).norm(2)
                    # if p is not p2:
                    dists.append(d.cpu().item())
                    kij = torch.exp(-(d**2) / h_kernel**2 / 2)
                    new_parameters[i][l] = (
                        ((new_parameters[i][l] + p2.grad.data) -
                            (d / h_kernel**2) * alpha) /
                        float(len(all_pgs))) * kij
    h_kernel = np.median(dists)
    h_kernel = np.sqrt(0.5 * h_kernel / np.log(len(all_pgs)) + 1)
    for i in range(len(all_pgs)):
        for l, p in enumerate(all_pgs[i].parameters()):
            if p.grad is not None:
                p.grad.data = new_parameters[i][l]

    return h_kernel    



def generate_freezed_particles(mdl , num_ensemble, device):

    classification_head = get_classification_head()
    image_encoder = ImageEncoder(mdl)
    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()

    NET = NET.to(device)
    particles = []
    for i in range(num_ensemble):
            particles.append(copy.deepcopy(NET))

    print(f'number of individual models: {len(particles)}')  
    
    return particles  

def generate_lora_particles(particles):

    exclude_regex = re.compile(r'^lora.*')
    model_module = importlib.import_module("model")
    # count_trainable_parameters_lora = model_module.count_trainable_parameters

    # print("\nEntered LP1WITHLORA:")
    Visualization(particles[0]).structure_graph()
    # trainable, delta, total = count_trainable_parameters_lora(particles[0], None)
    # print(f"[VIT-B/ENTERED MODEL] Trainable Params: {trainable} | Delta Params: {delta} | Total Params: {total}")

    print("\nActual Backbone Model:")
    Visualization(particles[0]).structure_graph()
    # trainable, delta, total = count_trainable_parameters_lora(particles[0], None)

    delta_models = []
    for i, particle in enumerate(particles):
     
        delta_models.append(LoraModel(backbone_model=particle, 
                            modified_modules=['c_fc', 'c_proj'],
                            lora_r=int(config('lora_rank'))))
        delta_models[i].log()
        delta_models[i].freeze_module(exclude=["deltas", "ln_final"], set_state_dict=True)
        delta_models[i].log()
    
    return delta_models

def train_model_wrap_cifar(particles, trainloaders, valloader, k, device, config):
    h_kernel = 0
    criterion = nn.CrossEntropyLoss()

    best_losses = [float('inf')] * len(particles)
    best_val_accuracy = [float('inf')] * len(particles)

    lr_list = [0.001, 0.0009, 0.0005, 0.00025, 0.0008]*50
    learning_rates = lr_list[:int(config('opt'))]

    optimizers = [optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=float(config('Weight_decay'))) for model, lr in zip(particles, learning_rates)]


    for epoch in range(int(config('num_epochs'))):
        
        accumulated_losses = [0.0] * len(particles)
        num_batches = len(next(iter(trainloaders)))

        for j,batches in enumerate(zip(*trainloaders)):
            inputs_list = [batch[0] for batch in batches]
            targets_list = [batch[1] for batch in batches]
            for i, (model, imgs, lbls) in enumerate(zip(particles, inputs_list, targets_list)):
                
                scheduler = cosine_lr(
                                            optimizers[i],
                                            learning_rates[i],
                                            int(config("warmup_length")),
                                            int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                        )

                step = (
                            i // int(config('num_grad_accumulation'))
                            + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                                            )
                imgs, labels = imgs.to(device), lbls.to(device)

                optimizers[i].zero_grad()

                logits = model.backbone_model(imgs)

                loss = criterion(logits, labels)
                loss.backward()
                accumulated_losses[i] += loss.item()
            print(f'\rProcessing batch {j+1}/{num_batches}', end='')
       
            h_kernel =update_gradiants(particles, h_kernel)

            for optimizer in optimizers:
                scheduler(step)
                optimizer.step()

        average_losses = [loss_sum / num_batches for loss_sum in accumulated_losses]
        print(" ")
        for i, avg_loss in enumerate(average_losses):
            print(f"Epoch {epoch}, Model {i}, Average Epoch Loss: {avg_loss}")

    
        with torch.no_grad():
            for i,model in enumerate(particles):

                correct = 0
                total = 0
                losses_eval, step2 = 0., 0.
                for img, lbls in valloader:
                    img, label = img.to(device), lbls.to(device)

                    logits = model.backbone_model(img)
                    loss_val = criterion(logits, label)
                    losses_eval += loss_val.item()
                    _, predicted = torch.max(logits, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    step2 += 1

                accuracy = correct / total
                loss_val_final = losses_eval / step2
                print(f'[Epoch: {epoch}], val_acc_{i}: {accuracy:.4f}, val_loss_{i}: {loss_val_final:.4f}')
                
                # 3. Save Models with Best Validation Loss
                model_idx = particles.index(model)
                if loss_val_final < best_losses[model_idx]:
                    best_losses[model_idx] = loss_val_final
                    best_val_accuracy[model_idx] = accuracy
                    best_epoch = epoch
                    best_model = copy.deepcopy(model.state_dict())

                    best_model_path = f"Model/best_model_{i}_{config('dataset_name')}_series_{k}.pt"
                    torch.save(best_model, best_model_path)
                    print(f'Best model {i} at epoch {best_epoch} has been saved')

    with open(f"Model/best_val_accuracy_{k}.txt", "w") as file:
    
    
        for i,accuracy in enumerate(best_val_accuracy):
            file.write(f"best val_acc for model {i} is {accuracy}\n")
    print('finished')        



def train_model_wrap_camelyon(particles, trainloaders, valloader, k, device, config):
    h_kernel = 0
    criterion = nn.CrossEntropyLoss()

    best_losses = [float('inf')] * len(particles)
    best_val_accuracy = [float('inf')] * len(particles)

    lr_list = [0.001, 0.0009, 0.0005, 0.00025, 0.0008]*50
    learning_rates = lr_list[:int(config('opt'))]

    optimizers = [optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=float(config('Weight_decay'))) for model, lr in zip(particles, learning_rates)]


    for epoch in range(int(config('num_epochs'))):
        
        accumulated_losses = [0.0] * len(particles)
        num_batches = len(next(iter(trainloaders)))

        for j,batches in enumerate(zip(*trainloaders)):
            inputs_list = [batch[0] for batch in batches]
            targets_list = [batch[1] for batch in batches]
            for i, (model, imgs, lbls) in enumerate(zip(particles, inputs_list, targets_list)):
                
                scheduler = cosine_lr(
                                            optimizers[i],
                                            learning_rates[i],
                                            int(config("warmup_length")),
                                            int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                        )

                step = (
                            i // int(config('num_grad_accumulation'))
                            + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                                            )
                imgs, labels = imgs.to(device), lbls.to(device)

                optimizers[i].zero_grad()

                logits = model.backbone_model(imgs)

                loss = criterion(logits, labels)
                loss.backward()
                accumulated_losses[i] += loss.item()
            print(f'\rProcessing batch {j+1}/{num_batches}', end='')
       
            h_kernel =update_gradiants(particles, h_kernel)

            for optimizer in optimizers:
                scheduler(step)
                optimizer.step()

        average_losses = [loss_sum / num_batches for loss_sum in accumulated_losses]
        print(" ")
        for i, avg_loss in enumerate(average_losses):
            print(f"Epoch {epoch}, Model {i}, Average Epoch Loss: {avg_loss}")

    
        with torch.no_grad():
            for i,model in enumerate(particles):

                correct = 0
                total = 0
                losses_eval, step2 = 0., 0.
                for img, lbls, _ in valloader:
                    img, label = img.to(device), lbls.to(device)

                    logits = model.backbone_model(img)
                    loss_val = criterion(logits, label)
                    losses_eval += loss_val.item()
                    _, predicted = torch.max(logits, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    step2 += 1

                accuracy = correct / total
                loss_val_final = losses_eval / step2
                print(f'[Epoch: {epoch}], val_acc_{i}: {accuracy:.4f}, val_loss_{i}: {loss_val_final:.4f}')
                
                # 3. Save Models with Best Validation Loss
                model_idx = particles.index(model)
                if loss_val_final < best_losses[model_idx]:
                    best_losses[model_idx] = loss_val_final
                    best_val_accuracy[model_idx] = accuracy
                    best_epoch = epoch
                    best_model = copy.deepcopy(model.state_dict())

                    best_model_path = f"Model/best_model_{i}_{config('dataset_name')}_series_{k}.pt"
                    torch.save(best_model, best_model_path)
                    print(f'Best model {i} at epoch {best_epoch} has been saved')

    with open(f"Model/best_val_accuracy_{k}.txt", "w") as file:
    
    
        for i,accuracy in enumerate(best_val_accuracy):
            file.write(f"best val_acc for model {i} is {accuracy}\n")
    print('finished')   



#-----------------------------------------------------------------------------------------------

def train_model_wrap_places(particles, trainloaders, valloader, k, config):
    h_kernel = 0
    criterion = nn.CrossEntropyLoss()

    best_losses = [float('inf')] * len(particles)
    best_val_accuracy = [float('inf')] * len(particles)

    lr_list = [0.001, 0.0009, 0.0005, 0.00025, 0.0008]*50
    learning_rates = lr_list[:int(config('opt'))]

    optimizers = [optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr) for model, lr in zip(particles, learning_rates)]


    for epoch in range(int(config('num_epochs'))):
        
        accumulated_losses = [0.0] * len(particles)
        num_batches = len(next(iter(trainloaders)))

        for j,batches in enumerate(zip(*trainloaders)):
            inputs_list = [batch["images"] for batch in batches]
            targets_list = [batch["labels"] for batch in batches]
            for i, (model, imgs, lbls) in enumerate(zip(particles, inputs_list, targets_list)):

                scheduler = cosine_lr(
                                            optimizers[i],
                                            learning_rates[i],
                                            int(config("warmup_length")),
                                            int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                        )

                step = (
                            i // int(config('num_grad_accumulation'))
                            + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                                            )

                imgs, labels = imgs.cuda(), lbls.cuda()

                optimizers[i].zero_grad()

                logits = model.backbone_model(imgs)
                labels = labels.squeeze(dim=1)

                loss = criterion(logits, labels)
                loss.backward()
                accumulated_losses[i] += loss.item()
            print(f'\rProcessing batch {j+1}/{num_batches}', end='')
            
            # h_kernel = update_gradiants(particles, h_kernel)

            for optimizer in optimizers:
                scheduler(step)
                optimizer.step()
        print(" ")
        average_losses = [loss_sum / num_batches for loss_sum in accumulated_losses]
        for i, avg_loss in enumerate(average_losses):
            print(f"Epoch {epoch}, Model {i}, Average Epoch Loss: {avg_loss}")

    
        with torch.no_grad():
            for i,model in enumerate(particles):

                correct = 0
                total = 0
                losses_eval, step2 = 0., 0.
                for img, lbls in valloader:
                    img, label = img.cuda(), lbls.cuda()

                    logits = model.backbone_model(img)
                    label = label.squeeze(dim=1)
                    loss_val = criterion(logits, label)
                    losses_eval += loss_val.item()
                    _, predicted = torch.max(logits, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    step2 += 1

                accuracy = correct / total
                loss_val_final = losses_eval / step2
                print(f'[Epoch: {epoch}], val_acc_{i}: {accuracy:.4f}, val_loss_{i}: {loss_val_final:.4f}')
                
                # 3. Save Models with Best Validation Loss
                model_idx = particles.index(model)
                if loss_val_final < best_losses[model_idx]:
                    best_losses[model_idx] = loss_val_final
                    best_val_accuracy[model_idx] = accuracy
                    best_epoch = epoch
                    best_model = copy.deepcopy(model.state_dict())

                    best_model_path = f"Model/best_model_{i}_{config('dataset_name')}_series_{k}.pt"
                    torch.save(best_model, best_model_path)
                    print(f'Best model {i} at epoch {best_epoch} has been saved')

    with open(f"Model/best_val_accuracy_{k}.txt", "w") as file:
    # Write each accuracy value to the file, one value per line
        for i,accuracy in enumerate(best_val_accuracy):
            file.write(f"best val_acc for model {i} is {accuracy}\n")
    print('finished')    
