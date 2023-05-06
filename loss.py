import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Boost_CE(adv_outputs, targets):
    adv_probs = F.softmax(adv_outputs, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_targets = torch.where(tmp1[:, -1] == targets, tmp1[:, -2], tmp1[:, -1])
    loss =  F.cross_entropy(adv_outputs, targets) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_targets)
    
    return loss

def TRADES_loss(inputs, adv_inputs, targets, model):
    outputs = model(inputs)
    adv_outputs = model(adv_inputs)
    sup_loss = F.cross_entropy(outputs, targets)
    nat_probs = F.softmax(outputs, dim=1)
    adv_probs = F.softmax(adv_outputs, dim=1)
    rob_loss = F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1).mean()
    
    return sup_loss, rob_loss

def TRADES_LS_loss(inputs, adv_inputs, targets, model, smoothing):
    LS_loss = LabelSmoothingCrossEntropy(smoothing)
    outputs = model(inputs)
    sup_loss = LS_loss(outputs, targets)
    
    adv_outputs = model(adv_inputs)
    nat_probs = F.softmax(outputs, dim=1)
    adv_probs = F.softmax(adv_outputs, dim=1)
    
    rob_loss = F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1).mean()
    
    return sup_loss, rob_loss

# mart
def MART_loss(inputs, adv_inputs, targets, model):
    outputs = model(inputs)
    adv_outputs = model(adv_inputs)
    adv_probs = F.softmax(adv_outputs, dim=1)
    nat_probs = F.softmax(outputs, dim=1)
    true_probs = torch.gather(nat_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
    sup_loss = Boost_CE(adv_outputs, targets)
    rob_loss = (F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1) * (1. - true_probs)).mean()
    
    return sup_loss, rob_loss

# Proposed loss
def ARoW_CE_loss(inputs, adv_inputs, targets, model):
    outputs = model(inputs)
    adv_outputs = model(adv_inputs)
    adv_probs = F.softmax(adv_outputs, dim=1)
    nat_probs = F.softmax(outputs, dim=1)
    true_probs = torch.gather(adv_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
    sup_loss = F.cross_entropy(outputs, targets)
    rob_loss = (F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1) * (1. - true_probs)).mean()
    
    return sup_loss, rob_loss

def ARoW_loss(inputs, adv_inputs, targets, model, smoothing):
    LS_loss = LabelSmoothingCrossEntropy(smoothing)
    outputs = model(inputs)
    adv_outputs = model(adv_inputs)
    adv_probs = F.softmax(adv_outputs, dim=1)
    nat_probs = F.softmax(outputs, dim=1)
    true_probs = torch.gather(adv_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
    sup_loss = LS_loss(outputs, targets)
    rob_loss = (F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1) * (1. - true_probs)).mean()
    
    return sup_loss, rob_loss



def CoW_loss(inputs, adv_inputs, targets, model, smoothing):
    LS_loss = LabelSmoothingCrossEntropy(smoothing)
    outputs = model(inputs)
    adv_outputs = model(adv_inputs)
    adv_probs = F.softmax(adv_outputs, dim=1)
    nat_probs = F.softmax(outputs, dim=1)
    true_probs = torch.gather(nat_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
    sup_loss = LS_loss(outputs, targets)
    rob_loss = (F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1) * true_probs).mean()
    
    return sup_loss, rob_loss


def HAT_loss(inputs, adv_inputs, targets, model, std_model):
    outputs = model(inputs)
    adv_outputs = model(adv_inputs)
    rob_loss = (F.kl_div((F.softmax(adv_outputs, dim=1) + 1e-12).log(), F.softmax(outputs, dim=1), reduction='none').sum(dim=1)).mean()
    sup_loss = F.cross_entropy(outputs, targets)
    helper_inputs = inputs + 2*(adv_inputs-inputs)
    with torch.no_grad():
        helper_targets = std_model(adv_inputs).argmax(dim=1).detach()
    help_loss = F.cross_entropy(model(helper_inputs), helper_targets)
    
    return sup_loss, rob_loss, help_loss

def GAIR_AT_loss(adv_inputs, targets, model, Kappa, num_steps):
    adv_outputs = model(adv_inputs)
    normalized_reweight = GAIR(num_steps, Kappa).cuda()
    sup_loss = F.cross_entropy(adv_outputs, targets, reduction='none')
    sup_loss = sup_loss.mul(normalized_reweight).mean()
    
    return sup_loss

def GAIR_TRADES_loss(inputs, adv_inputs, targets, model, Kappa, num_steps):
    outputs = model(inputs)
    adv_outputs = model(adv_inputs)
    normalized_reweight = GAIR(num_steps, Kappa).cuda()
    sup_loss = F.cross_entropy(inputs, targets, reduction='none')
    sup_loss = sup_loss.mul(normalized_reweight).mean()
    rob_loss = F.kl_div((F.softmax(adv_outputs, dim=1)+1e-12).log(), F.softmax(outputs, dim=1), reduction='none').sum(dim=1).mean()
    
    return sup_loss, rob_loss


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


def GAIR(num_steps, Kappa, Lambda=-1, func="Tanh"):
    # Weight assign
    if func == "Tanh":
        reweight = ((Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).tanh()+1)/2
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Sigmoid":
        reweight = (Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).sigmoid()
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Discrete":
        reweight = ((num_steps+1)-Kappa)/(num_steps+1)
        normalized_reweight = reweight * len(reweight) / reweight.sum()
            
    return normalized_reweight

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()