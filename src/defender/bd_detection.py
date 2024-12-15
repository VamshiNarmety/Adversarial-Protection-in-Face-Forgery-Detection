import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
activations_storage = {'clean':{}, 'test':{}}
def hold_fn(model, input, ouput, layer_name, mode):
    if layer_name not in activations_storage[mode]:
        activations_storage[mode][layer_name] = []
    activations_storage[mode][layer_name].append(ouput.cpu().detach().numpy())
    
    
def register(model, mode):
    model.pretrained_model.classifier.register_forward_hook(lambda m,i,o: hold_fn(m, i, o, 'last_layer', mode))
    
    
def collect_activations(model, data_loader, mode):
    activations_storage[mode].clear()
    model.eval()
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            model(inputs)
            
        
def compute_distributions(activations):
    distributions = {}
    for layer_name, activations_list in activations.items():
        activations_concat = np.concatenate(activations_list, axis=0)
        neuron_mean = np.mean(activations_concat, axis=0)
        neuron_std = np.std(activations_concat, axis=0)
        distributions[layer_name] = (neuron_mean, neuron_std)
    return distributions


def compute_nas(activation, mean, std, k=3):
    lower_bound = mean - k*std
    upper_bound = mean + k*std
    return (activation>=lower_bound) & (activation<=upper_bound)


def compute_threshold(nas_scores, rejection_rate):
    nas_scores_sorted = sorted(nas_scores)
    index = int((len(nas_scores)*rejection_rate)/100)
    return nas_scores_sorted[index]


def compute_nas_scores(model, data_loader, distributions, mode):
    nas_scores = []
    collect_activations(model, data_loader, mode)
    for layer_name, (mean, std) in distributions.items():
        activations_list = activations_storage[mode][layer_name]
        for input in activations_list:
            per_neuron_nas = compute_nas(input, mean, std)
            score_nas = np.mean(per_neuron_nas)
            nas_scores.append(score_nas)
    return nas_scores


def detect_backdoor(model, data_loader, distributions, threshold, mode):
    classifications = []
    collect_activations(model, data_loader, mode)
    for layer_name, (mean, std) in distributions.items():
        activations_list = activations_storage[mode][layer_name]
        for input in activations_list:
            per_neuron_nas = compute_nas(input, mean, std)
            score_nas = np.mean(per_neuron_nas)
            if score_nas >= threshold:
                classifications.append(0)
            else:
                classifications.append(1)
    return classifications


    



