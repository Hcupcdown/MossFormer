import os
import random
from os.path import join as opj

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def get_params(args):
    
    params    = {}
    args_ref  = vars(args)
    args_keys = vars(args).keys()

    for key in args_keys:
        if '__' in key:
            continue
        else:
            temp_params = args_ref[key]
            if type(temp_params) == dict:
                params.update(temp_params)
            else:
                params[key] = temp_params
                
    return params
    
def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)
            
def rescale_conv(conv, reference):
    std   = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale    

def seed_init(seed=100):

    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)  


def args_dict(args):
    """
    Get your arguments and make dictionary.
    If you add some arguments in the model, you should edit here also.
    """
    args.dataset = {'train':args.train,'val':args.val,'test':args.test}
    args.setting = {'sample_rate':args.sample_rate, 'segment':args.segment}

    args.ex_name = os.getcwd().replace('\\','/').split('/')[-1]

    return args

def torch_sisdr(reference, estimation):
    reference_energy = torch.sum(reference ** 2, dim=-1, keepdims=True)
    optimal_scaling = torch.sum(reference * estimation, dim=-1, keepdims=True) / reference_energy
    projection = optimal_scaling * reference
    noise = estimation - projection
    ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1)
    return torch.mean(10 * torch.log10(ratio))

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    

