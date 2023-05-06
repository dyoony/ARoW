import torch
import torch.nn as nn


class FGSM():

    def __init__(self, model, epsilon=4/255, target_mode=False):

        self.model = model
        self.epsilon = epsilon
        self.target_mode=target_mode
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images, labels):

        x = images.clone()

        x.requires_grad_()
        
        outputs = self.model(x)
        
        self.model.zero_grad()
        loss = self.criterion(outputs, labels)
        loss.backward()
        adv_d = self.epsilon * x.grad.sign()
        
        if self.target_mode:
            x = x - adv_d
        else:
            x = x + adv_d
            
        x = torch.clamp(x, min=0, max=1)
        
        return x.detach(), adv_d
