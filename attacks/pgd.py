import torch
import torch.nn as nn
import torch.nn.functional as F


class PGD_Linf():

    def __init__(self, model, epsilon=8/255, step_size=2/255, num_steps=10, random_start=True, target_mode= False, criterion= 'ce', bn_mode='eval', train=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def perturb(self, x_nat, targets=None):
        if self.bn_mode == 'eval':
            self.model.eval()
            
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            #self.model.zero_grad()
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "kl":
                loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat), dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "revkl":
                loss = self.criterion_kl(F.log_softmax(self.model(x_nat), dim=1), F.softmax(outputs, dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "js":
                nat_probs = F.softmax(self.model(x_nat), dim=1)
                adv_probs = F.softmax(outputs, dim=1)
                mean_probs = (nat_probs + adv_probs)/2
                loss =  (self.criterion_kl(mean_probs.log(), nat_probs) + self.criterion_kl(mean_probs.log(), adv_probs))/2
                grad = torch.autograd.grad(loss, [x_adv])[0]
                
            if self.target_mode:
                x_adv = x_adv - self.step_size * grad.sign()
            else:
                x_adv = x_adv + self.step_size * grad.sign()
            
            x_adv = torch.min(torch.max(x_adv, x_nat - self.epsilon), x_nat + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            d_adv = x_adv - x_nat
            
        if self.train:
            self.model.train()
        
        
        return x_adv, d_adv

# Geometry-aware projected gradient descent (GA-PGD)
class GA_PGD():
    
    
    def __init__(self, model, epsilon=8/255, step_size=2/255, num_steps=10, random_start=True, target_mode= False, criterion= 'ce', bn_mode='eval', train=True):
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        
    def perturb(self, x_nat, targets=None):
        if self.bn_mode == 'eval':
            self.model.eval()
            
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()
            
        #(model, data, target, epsilon, step_size, num_steps,loss_fn,category, rand_init):
        #model.eval()
        Kappa = torch.zeros(x_adv.size(0))
        
        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            pred = outputs.argmax(dim=1)
            for p in range(len(x_adv)):
                if pred[p] == targets[p]:
                    Kappa[p] += 1
            self.model.zero_grad()    
            if self.criterion == "kl":
                loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat), dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                loss.backward()
                grad = x_adv.grad
                
            x_adv = x_adv + self.step_size * grad.sign()
                
            x_adv = torch.min(torch.max(x_adv, x_nat - self.epsilon), x_nat + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            d_adv = x_adv - x_nat
            
        if self.train:
            self.model.train()
        
        return x_adv, Kappa

    

def GA_earlystop(model, data, target, step_size, epsilon, perturb_steps, tau, type, random, omega):
    # Based on code from https://github.com/zjfheart/Friendly-Adversarial-Training
    
    model.eval()
    K = perturb_steps
    count = 0

    output_target = []
    output_adv = []
    output_natural = []
    output_Kappa = []

    control = torch.zeros(len(target)).cuda()
    control += tau
    Kappa = torch.zeros(len(data)).cuda()

    if random == False:
        iter_adv = data.cuda().detach()
    else:

        if type == "fat_for_trades" :
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        if type == "fat" or "fat_for_mart":
            iter_adv = data.detach() + torch.empty_like(data).uniform_(-epsilon, epsilon).cuda().detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
    iter_clean_data = data.cuda().detach()
    iter_target = target.cuda().detach()
    output_iter_clean_data = model(data)

    while K > 0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        for idx in range(len(pred)):
            if pred[idx] != target[idx]:
                if control[idx]==0:
                    output_index.append(idx)
                else:
                    control[idx]-=1
                    iter_index.append(idx)
            else:
                # Update Kappa
                Kappa[idx] += 1
                iter_index.append(idx)

        if (len(output_index)!=0):
            if (len(output_target) == 0):
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
                output_Kappa = Kappa[output_index].reshape(-1).cuda()
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)
                output_Kappa = torch.cat((output_Kappa, Kappa[output_index].reshape(-1).cuda()), dim=0)

        model.zero_grad()
        with torch.enable_grad():
            if type == "fat" or type == "fat_for_mart":
                loss_adv = nn.CrossEntropyLoss()(output, iter_target)
            if type == "fat_for_trades":
                criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        if len(iter_index) != 0:
            Kappa = Kappa[iter_index]
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()
            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)
            count += len(iter_target)
        else:
            return output_adv, output_target, output_natural, count, output_Kappa
        K = K-1

    if (len(output_target) == 0):
        output_target = iter_target.reshape(-1).squeeze().cuda()
        output_adv = iter_adv.reshape(-1, 3, 32, 32).cuda()
        output_natural = iter_clean_data.reshape(-1, 3, 32, 32).cuda()
        output_Kappa = Kappa.reshape(-1).cuda()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0).cuda()
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 32, 32).cuda()),dim=0).cuda()
        output_Kappa = torch.cat((output_Kappa, Kappa.reshape(-1)),dim=0).squeeze().cuda()
    
    return output_adv, output_target, output_natural, count, output_Kappa    
    

class Const_PGD_Linf():

    def __init__(self, model, epsilon=8*4/255, step_size=4/255, num_steps=10, random_start=True, target_mode= False, criterion= 'ce', bn_mode='eval', train=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def perturb(self, x_nat, targets):
        if self.bn_mode == 'eval':
            self.model.eval()
            
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            _, outputs = self.model(x_adv)
            #self.model.zero_grad()
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                loss.backward()
                grad = x_adv.grad
            elif self.criterion == "kl":
                loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat)[1], dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "revkl":
                loss = self.criterion_kl(F.log_softmax(self.model(x_nat)[1], dim=1), F.softmax(outputs, dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "js":
                nat_probs = F.softmax(self.model(x_nat)[1], dim=1)
                adv_probs = F.softmax(outputs, dim=1)
                mean_probs = (nat_probs + adv_probs)/2
                loss =  (self.criterion_kl(mean_probs.log(), nat_probs) + self.criterion_kl(mean_probs.log(), adv_probs))/2
                grad = torch.autograd.grad(loss, [x_adv])[0]
            if self.target_mode:
                x_adv = x_adv - self.step_size * grad.sign()
            else:
                x_adv = x_adv + self.step_size * grad.sign()
            
            d_adv = torch.clamp(x_adv - x_nat, min=-self.epsilon, max=self.epsilon).detach()
            x_adv = torch.clamp(x_nat + d_adv, min=0, max=1).detach()
            
        if self.train:
            self.model.train()
        
        
        return x_adv, d_adv
    


class PGD_L2():
    def __init__(self, model, epsilon=20/255, step_size=4/255, num_steps=10, random_start=True, target_mode= False, criterion='ce', bn_mode='eval', train=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode= target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def perturb(self, x_nat, targets):
        
        if self.bn_mode == 'eval':
            self.model.eval()
        
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            
            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            self.model.zero_grad()
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                loss.backward()
                grad = x_adv.grad
            elif self.criterion == "kl":
                loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat), dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "revkl":
                loss = self.criterion_kl(F.log_softmax(self.model(x_nat), dim=1), F.softmax(outputs, dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
                
            grad_norm = grad.abs().pow(2).view(x_nat.shape[0], -1).sum(1).pow(1./2)
            grad_norm = grad_norm.view(x_nat.shape[0], 1, 1, 1).expand_as(x_nat)
            d_adv = grad/grad_norm
            
            if self.target_mode:
                x_adv= x_adv - self.step_size * d_adv
            else:
                x_adv= x_adv + self.step_size * d_adv
            
            d_adv = (x_adv - x_nat).view(x_nat.shape[0], -1).detach()
            d_adv = d_adv.view(x_nat.shape)
            d_adv = torch.clamp(d_adv, min=-self.epsilon, max=self.epsilon)
            
            x_adv = torch.clamp(x_nat + d_adv, min=0, max=1).detach()
            
        if self.train:
            self.model.train()
        
        return x_adv, d_adv


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()   
        

def js_div(p_logits, q_logits):
    """
    Function that measures JS divergence between target and output logits:
    """
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    p_prob = F.softmax(p_logits)
    q_prob = F.softmax(q_logits)
    mean_prob = (p_prob + q_prob)/2
    
    js_loss = (kl_loss(mean_probs.log() , p_prob) + kl_loss(mean_probs.log() , q_prob))/2
    return js_loss





'''
class PGD_Linf():

    def __init__(self, model, epsilon=8*4/255, step_size=4/255, num_steps=10, random_start=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images, labels):

        if self.random_start:
            x = images + (torch.rand_like(images) - 0.5) * 2 * self.epsilon
            x = torch.clamp(x, min=-2, max=2)
        else:
            x = images.clone()

        for _ in range(self.num_steps):

            x.requires_grad_()
            outputs = self.model(x)
            self.model.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            x= x+ self.step_size * x.grad.sign()
            adv_d= torch.clamp(x-images, min=-self.epsilon, max=self.epsilon).detach()
            
            x = torch.clamp(images + adv_d, min=-2, max=2)
        
        return x.detach(), adv_d


class PGD_L2():
    # [-2, 2] normalization => 4*episilon/255 means epsilon attck in [0, 1] normalization
    def __init__(self, model, epsilon=20*4/255, step_size=4/255, num_steps=10, random_start=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images, labels):

        if self.random_start:
            x = images + (torch.rand_like(images) - 0.5) * 2 * self.epsilon
            x = torch.clamp(x, min=-2, max=2)
        else:
            x = images.clone()

        for _ in range(self.num_steps):
            
            x.requires_grad_()
            outputs = self.model(x)
            self.model.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            grad=x.grad
            grad_norm = grad.abs().pow(2).view(x.shape[0], -1).sum(1).pow(1./2)
            grad_norm = grad_norm.view(x.shape[0], 1, 1, 1).expand_as(x)
            adv_d = grad/grad_norm
            
            x= x +self.step_size * adv_d
            
            adv_d= (x- images).view(x.shape[0], -1).detach()
            adv_d= adv_d.view(x.shape)
            adv_d= torch.clamp(adv_d, min=-self.epsilon, max=self.epsilon)
            
            x= torch.clamp(images + adv_d, min=-2, max=2)
            

        return x.detach(), adv_d
'''