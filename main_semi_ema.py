from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import shutil
import time
from datetime import date
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

#import models.wideresnet as wrn_models
import models.wideresnet as wrn_models
#import models.wideresnetwithswish as wrn_models
import models.resnet as res_models_plain
import models.resnet_silu as res_models
import models.preact_resnet_silu as pre_res_models

import load_data.datasets as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from load_data.datasets import SemiSupervisedDataset, SemiSupervisedSampler
from utils.swa import moving_average, bn_update
from utils.ema import ema_update, update_bn
from loss import *
from utils_awp import *
from attacks.pgd import PGD_Linf, GA_PGD
from attacks.earlystop import Early_PGD

parser = argparse.ArgumentParser(description='PyTorch Adversarially Robust Model')

# Semi-supervised training and configuration
parser.add_argument('--aux_data_filename', default='ti_500K_pseudo_labeled.pickle', choices=['ti_500K_pseudo_labeled.pickle', 'cifar10_ddpm.npz'], type=str, help='Path to pickle file containing undata and pseudo-labels used for RST')
parser.add_argument('--unsup_fraction', default=0.7, type=float,
                    help='Fraction of unlabeled examples in each batch; implicitly sets the weight of unlabeled data in the loss. If set to -1, batches are sampled from a single pool')
parser.add_argument('--aux_take_amount', default=None, type=int, help='Number of random aux examples to retain. None retains all aux data.')
parser.add_argument('--remove_pseudo_labels', action='store_true', default=False, help='Performs training without pseudo-labels (rVAT)')
parser.add_argument('--svhn_extra', action='store_true', default=False,
                    help='Adds the extra SVHN data')

########################## model setting ##########################
parser.add_argument('--depth', type=int, default=28, help='wideresnet depth factor')
parser.add_argument('--widen_factor', type=int, default=10, help='wideresnet widen factor')
parser.add_argument('--activation', type=str, default= 'SiLU', choices=['ReLU', 'LeakyReLU', 'SiLU'], help='choice of activation')
parser.add_argument('--model', type=str, default= 'wideresnet', help='architecture of model') #, choices=['resnet18, wideresnet'] : invalid choice

########################## optimization setting ##########################
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restayts)')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--wd', default=5e-4, type=float, metavar='WD', help='weight decay')
parser.add_argument('--lr_scheduler', type=str, default= 'Cosine', choices=['MultiStep', 'Cosine', 'Cyclic'], help='learning rate scheduling')
parser.add_argument('--grad_accumulation', type=int, default=8, help='number of steps for gradient accumulation')
#parser.add_argument('--eval_freq', default=5, type=int, metavar='N', help='frequency of evaluation')

######################### Checkpoints #############################
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

########################## basic setting ##########################
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_dir', default='/home/ydy0415/data/datasets', help='Directory of dataset')
parser.add_argument('--out', default='/home/ydy0415/data/experiments/arow/', help='Directory to output the result')
parser.add_argument('--tolerance', default=150, type=int, metavar='N', help='tolerance')

######################### Dataset #############################
parser.add_argument('--dataset', type=str, default= 'cifar10', choices=['cifar10', 'fmnist', 'svhn'], help='benchmark dataset')
parser.add_argument('--num_classes', type=int, default= 10, help='number of classes of benchmark dataset')

########################## attack setting ##########################
parser.add_argument('--train_attack', metavar='METHOD', default='pgd_linf', choices=['pgd_linf', 'gapgd_linf'], help=' attack method')
parser.add_argument('--perturb_loss', metavar='LOSS', default='kl', choices=['ce','kl','revkl','js'], help='perturbation loss for adversarial examples')
parser.add_argument('--eps', type=float, default=8, help= 'maximum of perturbation magnitude' )
parser.add_argument('--train_numsteps', type=int, default=10, help= 'train PGD number of steps')
parser.add_argument('--test_numsteps', type=int, default=10, help= 'test PGD number of steps')
parser.add_argument('--random_start', action='store_false', help='PGD use random start')
parser.add_argument('--bn_mode', metavar='BN', default='eval', choices=['eval', 'train'], help='batch normalization mode of attack')

########################## loss setting ##########################
parser.add_argument('--loss', metavar='LOSS', default='arow-ls', choices=['arow-ce', 'arow-ls', 'arowt-ls', 'cow', 'hat', 'hat-arow', 'mart', 'trades', 'trades-ls', 'madry' , 'fat-at', 'fat-trades', 'gair-at', 'gair-trades', 'fat-arow', 'trades-awp', 'arow-awp'], help='surrogate loss function to optimize')
parser.add_argument('--ls', type=float, default=0.2, help='alpha of label smoothing')
parser.add_argument('--lamb', type=float, default=6., help='coefficient of rob_loss')
parser.add_argument('--gamma', type=float, default=0.25, help='coefficient of hat_loss')

########################## EMA setting ##########################
parser.add_argument('--ema', action='store_true', help='ema usage flag (default: off)')
parser.add_argument('--ema_decay', type=float, default=0.995, help='ema usage flag (default: off)')

######################### add name #############################
parser.add_argument('--add_name', default='', type=str, help='add_name')

args = parser.parse_args()
print(args)

state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.set_device(args.gpu)
use_cuda = torch.cuda.is_available()


# Random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic=True

# To speed up training
torch.backends.cudnn.benchmark = True


if args.dataset in ['cifar10', 'svhn']:
    input_channel = 3
elif args.dataset in ['fmnist']:
    input_channel = 1

best_acc = 0  # best val accuracy
attack_best_acc = 0
tolerance = 0 # tolerance


test_acc = 0
attack_test_acc =0

def main():
    global best_acc
    global attack_best_acc
    global tolerance
    
    global test_acc
    global attack_test_acc
    
    args.out += str(args.dataset)
    args.out += '/' + str(date.today().strftime('%Y%m%d')[2:])
    
    out_directory = args.out + '/' + args.train_attack
    
    if args.model == "wideresnet":
        out_directory += '_wrn' + str(args.depth) + str(args.widen_factor)
        
    elif args.model == "resnet18":
        out_directory += '_resnet18'
    
    out_directory += '_loss' + str(args.loss) + '_perturbloss' + str(args.perturb_loss) + '_eps' + str(args.eps) + '_lrsche' + str(args.lr_scheduler) + '_epochs' + str(args.epochs)
    
    # first penalty   
    #choices=['arow-ce', 'arow-ls', 'cow', 'hat', 'hat-arow', 'mart', 'trades', 'trades-ls', 'madry' , 'fat-at', 'fat-trades', 'gair-at', 'gair-trades', 'fat-arow']
    if args.loss in ['trades', 'trades-ls', 'arow-ce', 'arow-ls', 'arowt-ls', 'cow', 'hat', 'mart', 'fat-trades', 'gair-trades', 'hat-arow', 'fat-arow']:
        out_directory += '_lamb' + str(args.lamb)
    
    if args.loss in ['arow-ls', 'arowt-ls', 'cow', 'trades-ls', 'fat-arow' ,'hat-arow']:
        out_directory += '_ls' + str(args.ls)
    
    # second penalty
    if args.loss in ['hat', 'hat-arow']:
        out_directory += '_gam' + str(args.gamma)    
    
    if args.ema:
        out_directory += '_ema' + str(args.ema)
    
    out_directory += '_seed' + str(args.seed)
        
    if args.add_name != '':
        out_directory +='_'+str(args.add_name)
    
    if not os.path.isdir(out_directory):
        mkdir_p(out_directory)
    
    result_png_path = os.path.join(out_directory, 'results.png')
    
    
    # Data
    print('==> Preparing ' + str(args.dataset))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    data_directory = args.data_dir
    
    trainset = SemiSupervisedDataset(base_dataset='cifar10',
                                 add_svhn_extra=args.svhn_extra,
                                 root=args.data_dir, train=True,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 add_aux_labels=not args.remove_pseudo_labels,
                                 aux_take_amount=args.aux_take_amount)
    
    #train_batch_sampler = SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, args.batch_size, args.unsup_fraction)
    train_batch_sampler = SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, args.batch_size, args.unsup_fraction, num_batches=int(np.ceil(50000 / args.batch_size)))
    train_loader = data.DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=0, pin_memory=True)
    
    
    testset   = SemiSupervisedDataset(base_dataset='cifar10', root=args.data_dir, train=False, download=True, transform=transform_val)
    test_loader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)
    
    
    trainset_eval = SemiSupervisedDataset(base_dataset='cifar10', add_svhn_extra=args.svhn_extra, root=args.data_dir, train=True, download=True, transform=transform_train)
    eval_train_loader = data.DataLoader(trainset_eval, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # Model
    
    def create_model(args):
        if args.model == 'wideresnet':
            print("==> creating WideResNet" + str(args.depth) + '-' + str(args.widen_factor))
            #model = wrn_models.wideresnetwithswish('wrn-28-10-swish', dataset=args.dataset, num_classes=args.num_classes).cuda(args.gpu)
            model = wrn_models.WideResNet(num_classes=10, depth=args.depth, widen_factor=args.widen_factor, activation=args.activation).cuda(args.gpu)
            if args.ema:
                ema_model = wrn_models.WideResNet(num_classes=10, depth=args.depth, widen_factor=args.widen_factor, activation=args.activation).cuda(args.gpu)
                #ema_model = wrn_models.wideresnetwithswish('wrn-28-10-swish', dataset=args.dataset, num_classes=args.num_classes).cuda(args.gpu)
            else:
                ema_model = None
                
            return model, ema_model
                
        elif args.model == 'resnet18':
            print("==> creating ResNet18")
            model = res_models.resnet('resnet18', input_channel, num_classes=10).cuda(args.gpu)
            if args.ema:
                ema_model = res_models.resnet('resnet18', input_channel, num_classes=10).cuda(args.gpu)
            else:
                ema_model = None
            
            return model, ema_model
                
        elif args.model == 'pre-resnet18':
            print("==> creating Pre-ResNet18")
            model = pre_res_models.preact_resnet('preact-resnet18', input_channel, num_classes=10).cuda(args.gpu)
            if args.ema:
                ema_model = pre_res_models.preact_resnet('preact-resnet18', input_channel, num_classes=10).cuda(args.gpu)
            else:
                ema_model = None
            
            return model, ema_model
        
    model, ema_model = create_model(args)
    
    if args.loss in ["trades-awp"]:
        proxy= create_model()
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=0.005)
    
    if args.loss in ["arow-awp"]:
        proxy= create_model()
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        awp_adversary = ArowAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=0.005)
    
    
    if args.loss in ['hat', 'hat-arow']:
        if args.model == 'wideresnet':
            std_model = wrn_models.WideResNet(num_classes=10, depth=args.depth, widen_factor=args.widen_factor, activation="ReLU").cuda(args.gpu)
            if args.loss == 'hat':
                checkpoint = torch.load(args.data_dir +  "/hat-cifar10/wrn2810.pth.tar", map_location='cuda:' + str(args.gpu))
            elif args.loss == 'hat-arow':
                checkpoint = torch.load(args.data_dir +  "/hat-cifar10/wrn2810.pth.tar", map_location='cuda:' + str(args.gpu))
        elif args.model == 'resnet18':
            std_model = res_models.resnet('resnet18', input_channel, num_classes=10).cuda(args.gpu)
            if args.loss == 'hat':
                checkpoint = torch.load(args.data_dir +  "/hat-cifar10/resnet18_plain.pth.tar", map_location='cuda:' + str(args.gpu))
            elif args.loss == 'hat-arow':
                checkpoint = torch.load(args.data_dir +  "/hat-cifar10/resnet18_ls0.2.pth.tar", map_location='cuda:' + str(args.gpu))
        std_model.load_state_dict(checkpoint['state_dict'])
        std_model.eval()
        del checkpoint
        torch.cuda.empty_cache()
        
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    criterion = nn.CrossEntropyLoss()
    
    if args.train_attack == 'pgd_linf':
        train_attack = PGD_Linf(model=model, epsilon=args.eps/255, step_size=(args.eps/4)/255, num_steps=args.train_numsteps, random_start=args.random_start,
                                criterion=args.perturb_loss, bn_mode = args.bn_mode, train = True)
    elif args.train_attack == 'gapgd_linf':
        train_attack = GA_PGD(model=model, epsilon=args.eps/255, step_size=(args.eps/4)/255, num_steps=args.train_numsteps, random_start=args.random_start,
                                criterion=args.perturb_loss, bn_mode = args.bn_mode, train = True)
        
    test_attack = PGD_Linf(model=ema_model, epsilon=args.eps/255, step_size=(args.eps/4)/255, num_steps=args.test_numsteps, random_start=args.random_start, criterion='ce', bn_mode = args.bn_mode, train = False)
        
    #no_decay = ['bias', 'bn']
    no_decay = ['bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=True)
    
    #ptimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    if args.lr_scheduler == "MultiStep":
        scheduler = lr_scheduler.MultiStepLR(optimizer , milestones=[2/4*args.epochs, 3/4*args.epochs], gamma=0.1) #cifar10
            
    elif args.lr_scheduler == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "Cyclic":
        #_NUM_SAMPLES = {'svhn': 73257, 'tiny-imagenet': 100000, 'imagenet100': 128334}
        num_samples = 50000
        update_steps = int(np.floor(num_samples/args.batch_size) + 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.2,
                                                        steps_per_epoch=update_steps, epochs=int(args.epochs))
     
    start_epoch = args.start_epoch

    # Resume
    title = args.dataset
    
    logger = Logger(os.path.join(out_directory, 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Train Loss', 'Test Loss', 'Test Acc.', 'Attack Loss' , 'Attack Acc.'])
    
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):

        print('\n'+args.train_attack +' Epoch: [%d | %d] LR: %.5f Tol: %d Best ts acc: %.2f Best_att_acc: %.2f ' % (epoch, args.epochs, optimizer.param_groups[0]['lr'], tolerance, best_acc, attack_best_acc))
        
        if args.loss in ["hat", 'hat-arow']:
            train_loss = train(train_loader, epoch, model, optimizer, use_cuda, attack=train_attack, std_model=std_model, ema_model=ema_model, scheduler=scheduler)
        elif args.loss in ["trades-awp", "arow-awp"]:
            train_loss = train(train_loader, epoch, model, optimizer, use_cuda, attack=train_attack, std_model=None, awp_adversary=awp_adversary, scheduler=scheduler)
        else:
            train_loss = train(train_loader, epoch, model, optimizer, use_cuda, attack=train_attack, ema_model=ema_model, scheduler=scheduler)
        
        update_bn(ema_model, model)
        
        if (epoch <= 400  and epoch % 5 == 0) or (epoch > 400 and epoch % 2 == 0):
            test_loss, test_acc = validate(test_loader, ema_model, criterion, use_cuda, mode='Test')
            attack_test_loss, attack_test_acc = validate(test_loader, ema_model, criterion, use_cuda, mode='Attack_test', attack=test_attack)
            #attack_test_loss, attack_test_acc = validate(test_loader, model, criterion, use_cuda, mode='Attack_test', attack=test_attack)
            logger.append([round(epoch), train_loss, test_loss, test_acc, attack_test_loss,  attack_test_acc])
            
        if args.lr_scheduler in ["MultiStep", "Cosine"]:
            scheduler.step()
            
        # save model
        is_attack_best = attack_test_acc > attack_best_acc
        attack_best_acc = max(attack_test_acc, attack_best_acc)
        
        is_best = test_acc >= best_acc
        best_acc = max(test_acc, best_acc)
        
        if is_best:
            best_acc = test_acc
        if is_attack_best:
            attack_best_acc = attack_test_acc
        
        if args.ema:
            if  is_attack_best:
                save_checkpoint(out_directory, epoch,
                filename='robust_best.pth.tar',
                ema_state_dict = ema_model.state_dict(),
                state_dict = model.state_dict(),
                test_acc =  test_acc,
                attack_test_acc = attack_test_acc,
                optimizer = optimizer.state_dict()
                )
            elif epoch % 10 == 0:
                save_checkpoint(out_directory, epoch,
                filename=str(epoch) + '_model.pth.tar',
                ema_state_dict = ema_model.state_dict(),
                test_acc =  test_acc,
                attack_test_acc = attack_test_acc,
                optimizer = optimizer.state_dict()
                )
                
            elif epoch == args.epochs:
                save_checkpoint(out_directory, epoch, 
                filename='last.pth.tar',
                ema_state_dict = ema_model.state_dict(),
                test_acc = test_acc,
                attack_test_acc = attack_test_acc
                )
                
        elif not args.ema:
            if is_attack_best:
                save_checkpoint(out_directory, epoch, 
                    filename='robust_best.pth.tar',       
                    state_dict = model.state_dict(),
                    test_acc = test_acc,
                    attack_test_acc = attack_test_acc,
                    optimizer = optimizer.state_dict()
                    )
            
            if epoch == args.epochs:
                save_checkpoint(out_directory, epoch, 
                filename='last.pth.tar',
                state_dict = model.state_dict(),
                test_acc = test_acc,
                attack_test_acc = attack_test_acc
                )


        if is_attack_best:
            tolerance = 0
        else:
            tolerance += 1

    logger.close()

    print('Best test acc:')
    print(best_acc)

    print('Best attack acc:')
    print(attack_best_acc)

def train(train_loader, epoch, model, optimizer, use_cuda, attack, ema_model, std_model=None, awp_adversary=None, scheduler=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    sup_losses = AverageMeter()
    reg_losses = AverageMeter()
    losses = AverageMeter()
    
    
    ce_loss=nn.CrossEntropyLoss()
    end = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    bar = Bar('{:>12}'.format('Training'), max=len(train_loader))
    
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)
        batch_size = inputs.size(0)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
    
        with torch.cuda.amp.autocast():
            
            if args.loss == "arow-ce":
                if args.perturb_loss not in ("kl", "revkl", "js"):
                     raise ValueError("perturb loss must be kl or revkl divergence.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = ARoW_CE_loss(inputs, adv_inputs, targets, model)
                reg_loss = args.lamb * reg_loss
                loss= sup_loss + reg_loss
                
            elif args.loss == "arow-ls":
                #if args.perturb_loss not in ("kl", "revkl", "js"):
                #     raise ValueError("perturb loss must be kl or revkl divergence.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = ARoW_loss(inputs, adv_inputs, targets, model, args.ls)
                reg_loss = args.lamb * reg_loss
                loss = sup_loss + reg_loss
            
            elif args.loss == "arowt-ls":
                if args.perturb_loss not in ("kl", "revkl", "js"):
                     raise ValueError("perturb loss must be kl or revkl divergence.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = ARoWT_loss(inputs, adv_inputs, targets, model, args.ls)
                reg_loss = args.lamb * reg_loss
                loss = sup_loss + reg_loss
            
            elif args.loss == "cow":
                if args.perturb_loss not in ("kl", "revkl", "js"):
                     raise ValueError("perturb loss must be kl or revkl divergence.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = CoW_loss(inputs, adv_inputs, targets, model, args.ls)
                reg_loss = args.lamb * reg_loss
                loss = sup_loss + reg_loss
                
            elif args.loss == "hat":
                if args.perturb_loss not in ("kl", "revkl", "js"):
                     raise ValueError("perturb loss must be kl or revkl divergence.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, rob_loss, hat_loss = HAT_loss(inputs, adv_inputs, targets, model, std_model)
                reg_loss = args.lamb * rob_loss + args.gamma* hat_loss
                loss = sup_loss + reg_loss
                
            elif args.loss == "hat-arow":
                if args.perturb_loss not in ("kl", "revkl", "js"):
                     raise ValueError("perturb loss must be kl or revkl divergence.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                #CoWHAT_loss(inputs, adv_inputs, targets, model, std_model, smoothing)
                sup_loss, rob_loss, hat_loss = HAT_ARoW_loss(inputs, adv_inputs, targets, model, std_model, args.ls)
                reg_loss = args.lamb * rob_loss + args.gamma* hat_loss
                loss = sup_loss + reg_loss
            
            elif args.loss == "mart":
                if args.perturb_loss != "ce":
                     raise ValueError("perturb loss must be ce.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = MART_loss(inputs, adv_inputs, targets, model)
                reg_loss = args.lamb * reg_loss
                loss= sup_loss + reg_loss
                
            elif args.loss == "trades":
                if args.perturb_loss not in ("kl", "revkl", "js"):
                     raise ValueError("perturb loss must be kl or revkl divergence.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = TRADES_loss(inputs, adv_inputs, targets, model)
                reg_loss = args.lamb*reg_loss
                loss = sup_loss + reg_loss
            
            elif args.loss == "trades-ls":
                if args.perturb_loss not in ("kl", "revkl", "js"):
                     raise ValueError("perturb loss must be kl or revkl divergence.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = TRADES_LS_loss(inputs, adv_inputs, targets, model)
                reg_loss = args.lamb*reg_loss
                loss = sup_loss + reg_loss
            
            elif args.loss == "madry":
                if args.perturb_loss != "ce":
                     raise ValueError("perturb loss must be ce.")
                adv_inputs, _ = attack.perturb(inputs, targets)
                adv_outputs= model(adv_inputs)
                sup_loss = ce_loss(adv_outputs, targets)
                reg_loss = torch.tensor(0)
                loss = sup_loss
            
            elif args.loss == "gair-at":
                if args.perturb_loss != "ce":
                        raise ValueError("perturb loss must be ce.")
                if epoch < 50:
                    adv_inputs, _ = attack.perturb(inputs, targets)
                    adv_outputs= model(adv_inputs)
                    sup_loss = ce_loss(adv_outputs, targets)
                    reg_loss = torch.tensor(0)
                    loss = sup_loss
                else:
                    adv_inputs, Kappa = attack.perturb(inputs, targets)
                    sup_loss = GAIR_AT_loss(adv_inputs, targets, model, Kappa, args.train_numsteps)
                    reg_loss = torch.tensor(0)
                    loss = sup_loss
            
            elif args.loss == "gair-trades":
                if args.perturb_loss != "kl":
                     raise ValueError("perturb loss must be kl.")
                adv_inputs, Kappa = attack.perturb(inputs, targets)
                sup_loss, reg_loss = GAIR_TRADES_loss(inputs, adv_inputs, targets, model, Kappa, args.train_numsteps)
                reg_loss = args.lamb*reg_loss
                loss = sup_loss + reg_loss
            
            elif args.loss == "fat-at":
                if args.perturb_loss != "ce":
                     raise ValueError("perturb loss must be ce.")
                
                # def Early_PGD(model, data, target, step_size, epsilon, perturb_steps, tau,loss_fn, rand_init=True, omega=0):
                adv_inputs, targets, _, _ = Early_PGD(model=model, data=inputs, target=targets, step_size = (args.eps/4)/255, epsilon = args.eps/255, perturb_steps = args.train_numsteps, tau=5, loss_fn =args.perturb_loss, rand_init =args.random_start, omega=0)
                model.train()
                adv_outputs= model(adv_inputs)
                sup_loss = ce_loss(adv_outputs, targets)
                reg_loss = torch.tensor(0)
                reg_loss = args.lamb*reg_loss
                loss = sup_loss + reg_loss
                
            elif args.loss == "fat-trades":
                if args.perturb_loss != "kl":
                     raise ValueError("perturb loss must be kl.")
                adv_inputs, targets, inputs, _ = Early_PGD(model=model, data=inputs, target=targets, step_size = (args.eps/4)/255, epsilon = args.eps/255, perturb_steps = args.train_numsteps, tau=5, loss_fn =args.perturb_loss, rand_init =args.random_start, omega=0)
                
                model.train()
                sup_loss, reg_loss = TRADES_loss(inputs, adv_inputs, targets, model)
                reg_loss = args.lamb*reg_loss
                loss = sup_loss + reg_loss
            
            elif args.loss == "fat-arow":
                if args.perturb_loss != "kl":
                     raise ValueError("perturb loss must be kl.")
                adv_inputs, targets, inputs, _ = Early_PGD(model=model, data=inputs, target=targets, step_size = (args.eps/4)/255, epsilon = args.eps/255, perturb_steps = args.train_numsteps, tau=2, loss_fn =args.perturb_loss, rand_init =args.random_start, omega=0)
                
                model.train()
                sup_loss, reg_loss = ARoW_loss(inputs, adv_inputs, targets, model, args.ls)
                reg_loss = args.lamb * reg_loss
                loss = sup_loss + reg_loss
                
            elif args.loss == "trades-awp":
                if args.perturb_loss != "kl":
                    raise ValueError("perturb loss must be kl.")
                if epoch >= 10:
                    adv_inputs, _ = attack.perturb(inputs, targets)
                    awp = awp_adversary.calc_awp(inputs_adv=adv_inputs,
                                         inputs_clean=inputs,
                                         targets=targets,
                                         beta=args.lamb)
                    awp_adversary.perturb(awp)    
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = TRADES_loss(inputs, adv_inputs, targets, model)
                reg_loss = args.lamb*reg_loss
                loss = sup_loss + reg_loss
            
            elif args.loss == "arow-awp":
                if args.perturb_loss != "kl":
                    raise ValueError("perturb loss must be kl.")
                if epoch >= 10:
                    adv_inputs, _ = attack.perturb(inputs, targets)
                    awp = awp_adversary.calc_awp(inputs_adv=adv_inputs,
                                         inputs_clean=inputs,
                                         targets=targets,
                                         beta=args.lamb,
                                         smoothing=args.ls
                                                )
                    awp_adversary.perturb(awp)
                adv_inputs, _ = attack.perturb(inputs, targets)
                sup_loss, reg_loss = ARoW_loss(inputs, adv_inputs, targets, model, args.ls)
                reg_loss = args.lamb * reg_loss
                loss = sup_loss + reg_loss

        # record loss
        sup_losses.update(sup_loss.item(), inputs.size(0))
        reg_losses.update(reg_loss.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

         # compute gradient and do SGD step
        loss = loss / args.grad_accumulation
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % args.grad_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema_update(ema_model, model, decay_rate=args.ema_decay)
            
        

        if args.lr_scheduler in ["Cyclic"]:
            scheduler.step()
        
        if epoch >= 10 and args.loss in ['trades-awp', 'arow-awp']:
            awp_adversary.restore(awp)
        
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Sup_loss: {sup_loss:.4f} | Reg_loss: {reg_loss:.4f} |  Tot loss:{loss:.4f}'.format(
                    batch   = batch_idx + 1,
                    size    = len(train_loader),
                    data    = data_time.avg,
                    bt      = batch_time.avg,
                    total   = bar.elapsed_td,
                    eta     = bar.eta_td,
                    sup_loss=sup_losses.avg,
                    reg_loss=reg_losses.avg,
                    loss=losses.avg
                    )
        bar.next()
    bar.finish()
                  
    return losses.avg

def validate(val_loader, model, criterion, use_cuda, mode, attack=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    bar = Bar('{mode:>12}'.format(mode=mode), max=len(val_loader))
   
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        
        # compute output
        if attack is not None:
            adv_inputs, _ = attack.perturb(inputs, targets)
            outputs = model(adv_inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
        bar.next()
    bar.finish()
        
    return (losses.avg, top1.avg)


def save_checkpoint(out_dir, epoch, filename='checkpoint.pth.tar', **kwargs):
    state={
        'epoch' : epoch
    }
    state.update(kwargs)
    filepath = os.path.join(out_dir, filename)
    torch.save(state, filepath)
    
    print("==> saving best model")



if __name__ == '__main__':
    main()