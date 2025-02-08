import os
import json
import torch
import copy
import numpy as np
from pathlib import Path
from decouple import config
import matplotlib.pyplot as plt
from decouple import config
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from torchmetrics.classification import CalibrationError
from torchmetrics.classification import MulticlassAUROC
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

class Paths():
    def __init__(self, config):
        self.path_results = ''
        self.path_store_model = ''
        self.path_figure=''

    def create_path(self):
        """
        This function creates a path for saving the best models and results
        :param settings: settings of the project
        :returns:
            path_results: the path for saving results
            path_saved_models: the path for saving the trained models
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.path_results = dir_path + '/Results/' 
        self.path_data = dir_path + '/Data/'
        self.path_store_model = dir_path +'/' + config('model_path') 
        # self.path_store_model = dir_path +'/' + '/Models/'
        self.path_figure = self.path_results + '/Figures/'
        self.path_checkpoints = self.path_store_model + '/losses/'

        Path(self.path_results).mkdir(parents=True, exist_ok=True)
        Path(self.path_figure).mkdir(parents=True, exist_ok=True)
        Path(self.path_store_model).mkdir(parents=True, exist_ok=True)
        Path(self.path_data).mkdir(parents=True, exist_ok=True)
        Path(self.path_checkpoints).mkdir(parents=True, exist_ok=True)



def add_noise_to_parameters(model, noise_std):
    device = next(model.parameters()).device  
    for param in model.parameters():
        noise = torch.randn(param.size(), device=device) * noise_std  
        param.data.add_(noise)



def generate_and_save_plot(loss_values, val_loss_values, save_path):

    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()



def calculate_metrics(logits, labels, num_classes, n_bins=15):

    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    probs = F.softmax(logits, dim=1)

    if config('dataset_name').upper() == "CAMELYON":
        task_type = 'binary'
    else: 
        task_type = 'multiclass'  

    ece_metric = CalibrationError(n_bins=n_bins, norm='l1', task=task_type, num_classes=num_classes)
    mce_metric = CalibrationError(n_bins=n_bins, norm='max', task=task_type, num_classes=num_classes)


    # Calculate metrics
    ece = ece_metric(probs, labels)
    mce = mce_metric(probs, labels)


    return ece.item(), mce.item()


def generate_results(all_scores, all_labels, noise, i, paths):

    accuracy = accuracy_score(all_labels, all_scores)
    print(f"Accuracy: {accuracy}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_scores)
    print("Confusion Matrix:")
    print(conf_matrix)


    # Calculate evaluation metrics
    f1 = f1_score(all_labels, all_scores, average='weighted')
    precision = precision_score(all_labels, all_scores, average='weighted')
    recall = recall_score(all_labels, all_scores, average='weighted')

    # Prepare data for the bar plot
    print(f"results for model with {noise} noise: f1-score: {f1}, precision: {precision}, recall: {recall}")
    fil_path = paths + f'evaluation_model_{i}_noise_{noise}_metrics.txt'  

    with open(fil_path, 'w') as file:
        file.write('F1 Score: {:.4f}\n'.format(f1))
        file.write('Precision: {:.4f}\n'.format(precision))
        file.write('Recall: {:.4f}\n'.format(recall))
        file.write('Accuracy: {:.4f}\n'.format(accuracy))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix noise: {noise}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

    # Save confusion matrix 
    plt.savefig(paths + f'noise_{noise}_confusion_matrix_model_{i}.png')
    plt.close()

    #np.save(paths + f'noise_{noise}_scores.npy', all_scores)  

def bar_plot_diff(data, noise, path):
    keys = list(data.keys())
    values = list(data.values())

    plt.bar(keys, values)
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title(f'Bar Plot noise: {noise}')
    plt.xticks(rotation=45)
    plt.savefig(path + f'model_noise_{noise}_barplot.png')
    plt.close()


def block_diff(model, model_new):
    my_dict = {}

    blocks_to_compare = ['visual', 'token_embedding', 'ln_final']

    for block_name in blocks_to_compare:
        pretrained_block = dict(model.named_children())[block_name]
        fine_tuned_block = dict(model_new.named_children())[block_name]
        
        param_names =[]
        param_diff_f = []
        for (pretrained_param_name, pretrained_param), (fine_tuned_param_name, fine_tuned_param) in zip(pretrained_block.named_parameters(), fine_tuned_block.named_parameters()):
            param_diff = torch.norm(pretrained_param - fine_tuned_param)  
            
            param_names.append(pretrained_param_name)
            pp = param_diff.detach().cpu().numpy().item()
            param_diff_f.append(pp)

        
        my_dict[block_name] = sum(param_diff_f)

    return      my_dict   


def generate_equal_subsets(dataset, num_splits):
    torch.manual_seed(42)

    labels = torch.tensor(dataset.targets)
    splitter = StratifiedShuffleSplit(n_splits=num_splits, test_size=None, random_state=42)
    subset_indices = list(splitter.split(torch.zeros_like(labels), labels))

    subsets = []
    for indices in subset_indices:
        subset = torch.utils.data.Subset(dataset, indices[0])
        subsets.append(subset)

    return subsets


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster



def generate_particles(model , num_ensemble):
    particles = []
    for i in range(num_ensemble):
            particles.append(copy.deepcopy(model))

    print(f'number of individual models: {len(particles)}')  
    
    return particles      

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster

