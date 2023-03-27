import random
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MetricsStorage(object):
    """Stores prediction metrics for training and validation"""

    def __init__(self):
        self.metrics = {
            'train_loss': AverageMeter(),
            'train_acc': AverageMeter(),
            'train_rec': AverageMeter(),
            'train_prec': AverageMeter(),
            'train_f1': AverageMeter(),
            'val_loss': AverageMeter(),
            'val_acc': AverageMeter(),
            'val_rec': AverageMeter(),
            'val_prec': AverageMeter(),
            'val_f1': AverageMeter()
        }
        pass
    
    def update(self, mode, loss, size, preds, labels, last_batch):
        acc = accuracy_score(preds, labels)
        recall = recall_score(preds, labels, average='macro', zero_division=0)
        precision = precision_score(preds, labels, average='macro', zero_division=0)
        f1 = f1_score(preds, labels, average='macro', zero_division=0)

        if mode == 'training':
            self.metrics['train_loss'].update(loss, size, last_batch)
            self.metrics['train_acc'].update(acc, size, last_batch)
            self.metrics['train_rec'].update(recall, size, last_batch)
            self.metrics['train_prec'].update(precision, size, last_batch)
            self.metrics['train_f1'].update(f1, size, last_batch)
        else:
            self.metrics['val_loss'].update(loss, size, last_batch)
            self.metrics['val_acc'].update(acc, size, last_batch)
            self.metrics['val_rec'].update(recall, size, last_batch)
            self.metrics['val_prec'].update(precision, size, last_batch)
            self.metrics['val_f1'].update(f1, size, last_batch)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.all_values_avg = []

    def update(self, val, n, last_batch):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if last_batch:
            self.all_values_avg += [self.avg]
            
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()
        
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()

def get_scheduler(optimizer, args):

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    gamma=0.1,
                                                    step_size=150)

    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=50)

    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epochs, 
                                                               eta_min=args.lr * 1e-3)

def load_model(path, device):
    model = torch.load(path, map_location=device)
    return model
                                                               