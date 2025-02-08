import os
# import open_clip
import torch
from tqdm import tqdm
from src.modeling import ClassificationHead, ImageEncoder
from decouple import config


def get_classification_head():

    dataset_name = config('dataset_name')
    filename = "./head/"+f"head_{dataset_name}.pt"
    
    print(f"Classification head for {dataset_name} has been loaded")

    return ClassificationHead.load(filename)

