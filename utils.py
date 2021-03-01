import shutil

import numpy as np
from sklearn.metrics import accuracy_score

import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def cal_acc(gt_list, predict_list, num):
    acc_sum = 0
    for n in range(num):
        y = []
        pred_y = []
        for i in range(len(gt_list)):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt)
                pred_y.append(predict)
        print ('{}: {:4f}'.format(n if n != (num - 1) else 'Unk', accuracy_score(y, pred_y)))
        if n == (num - 1):
            print ('Known Avg Acc OS*: {:4f}'.format(acc_sum / (num - 1)))
        acc_sum += accuracy_score(y, pred_y)
    print ('Avg Acc OS: {:4f}'.format(acc_sum / num))
    print ('Overall Acc : {:4f}'.format(accuracy_score(gt_list, predict_list)))
    return acc_sum / num
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


import os
from torch.autograd import Variable
import logging
import torch.nn.functional as F
import torch.nn as nn
import shutil
import torch
from datetime import datetime
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import math
import numpy as np

def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(dist.new_group(ranks=rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank//group_size]

def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[line:%(lineno)4d] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

def accuracy_ep(output, target, epsilon, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred_output, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred_output = pred_output.t()

    pred[0][pred_output[0] < (1./10. + epsilon)] = 10

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(correct)
    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):

        # compute the negative likelyhood
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred_output, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred_output = pred_output.t()


    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(correct)
    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_2(output, output_uk, target, epsilon, topk=(1,), unknown=12):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred_output, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred_output = pred_output.t()
    pred_uk = F.sigmoid(output_uk)
    pred[0][torch.squeeze(pred_uk>epsilon)] = unknown

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




class IterLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, last_iter=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestone, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.last_iter = last_iter

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.last_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr']*self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr


class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        #return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + 'checkpoint.pth.tar', filename + 'model_best.pth.tar')

def load_state(path, model, optimizer=None):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        ckpt_keys = set(checkpoint['state_dict'].keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))

        if optimizer != None:
            best_prec1 = checkpoint['best_prec1']
            last_iter = checkpoint['step']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (iter {})"
                  .format(path, last_iter))
            return best_prec1, last_iter
    else:
        print("=> no checkpoint found at '{}'".format(path))




class PredictionEvaluator_ep(object):
    def __init__(self, y, n_classes):
        self.y = y
        self.n_classes = n_classes
        self.hist = np.bincount(y, minlength=self.n_classes)

    def evaluate(self, tgt_pred_prob_y, epsilon):
        tgt_pred_y = np.argmax(tgt_pred_prob_y, axis=1)
        tgt_pred = np.max(tgt_pred_prob_y, axis = 1)

        tgt_pred_y[tgt_pred < (1./10. + epsilon)] = 10

        aug_class_true_pos = np.zeros((self.n_classes,))

        # Compute per-class accuracy
        for cls_i in range(self.n_classes):
            aug_class_true_pos[cls_i] = ((self.y == cls_i) & (tgt_pred_y == cls_i)).sum()

        aug_cls_acc = aug_class_true_pos.astype(float) / np.maximum(self.hist.astype(float), 1.0)

        mean_aug_class_acc = aug_cls_acc.mean()


        return mean_aug_class_acc, aug_cls_acc
def sigmoid(x):
    return 1 / (1+np.exp(-x))
class PredictionEvaluator_2(object):
    def __init__(self, y, n_classes):
        self.y = y
        self.n_classes = n_classes
        self.hist = np.bincount(y, minlength=self.n_classes)

    def evaluate(self, tgt_pred_prob_y, uk_pred, epsilon):
        tgt_pred_y = np.argmax(tgt_pred_prob_y, axis=1)
        tgt_pred = np.max(tgt_pred_prob_y, axis = 1)
        uk_pred = sigmoid(uk_pred)
        tgt_pred_y[np.squeeze(uk_pred > epsilon)] = self.n_classes

        aug_class_true_pos = np.zeros((self.n_classes+1,))

        # Compute per-class accuracy
        for cls_i in range(self.n_classes+1):
            aug_class_true_pos[cls_i] = ((self.y == cls_i) & (tgt_pred_y == cls_i)).sum()

        aug_cls_acc = aug_class_true_pos.astype(float) / np.maximum(self.hist.astype(float), 1.0)

        mean_aug_class_acc = aug_cls_acc.mean()


        return mean_aug_class_acc, aug_cls_acc



class PredictionEvaluator(object):
    def __init__(self, y, n_classes):
        self.y = y
        self.n_classes = n_classes
        self.hist = np.bincount(y, minlength=self.n_classes)

    def evaluate(self, tgt_pred_prob_y):
        tgt_pred_y = np.argmax(tgt_pred_prob_y, axis=1)
        tgt_pred = np.max(tgt_pred_prob_y, axis = 1)

        aug_class_true_pos = np.zeros((self.n_classes,))

        # Compute per-class accuracy
        for cls_i in range(self.n_classes):
            aug_class_true_pos[cls_i] = ((self.y == cls_i) & (tgt_pred_y == cls_i)).sum()

        aug_cls_acc = aug_class_true_pos.astype(float) / np.maximum(self.hist.astype(float), 1.0)

        mean_aug_class_acc = aug_cls_acc.mean()


        return mean_aug_class_acc, aug_cls_acc


class WeightEMA(object):
    def __init__(self, params, src_params, alpha=0.999):
        '''
        Will the src_param change?
        '''
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)


def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def bugged_cls_bal_bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def log_cls_bal(pred, tgt):
    return -torch.log(pred + 1.0e-6)


def compute_aug_loss(stu_out, tea_out, confidence_thresh, cls_balance=0.001, args=None):
    #need add softmax
    #need to add loss weight
    stu_out = F.softmax(stu_out, dim = 1)
    tea_out = F.softmax(tea_out, dim = 1)
    conf_tea = torch.max(tea_out, 1)[0]
    conf_mask = torch.gt(conf_tea, confidence_thresh).float()

    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss
    aug_loss = torch.mean(aug_loss, 1) * conf_mask
    if cls_balance > 0.0:
        avg_cls_prob = torch.mean(stu_out, 0)
        equalise_cls_loss = bugged_cls_bal_bce(
                            avg_cls_prob, float(1.0 / args.num_classes))
        equalise_cls_loss = torch.mean(equalise_cls_loss) * args.num_classes
        equalise_cls_loss = equalise_cls_loss * torch.mean(conf_mask, 0)
    else:
        equalise_cls_loss = None
    return aug_loss, conf_mask, equalise_cls_loss



def compute_aug_loss_double(stu_out, tea_out, confidence_thresh, cls_balance=0.001, args=None):
    #need add softmax
    #need to add loss weight
    stu_out = F.softmax(stu_out, dim = 1)
    tea_out = F.softmax(tea_out, dim = 1)
    conf_tea = torch.max(tea_out, 1)[0]
    conf_mask = torch.gt(conf_tea, confidence_thresh).float()

    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss * torch.exp(tea_out)
    aug_loss = torch.mean(aug_loss, 1) * conf_mask
    if cls_balance > 0.0:
        avg_cls_prob = torch.mean(stu_out, 0)
        equalise_cls_loss = bugged_cls_bal_bce(
                            avg_cls_prob, float(1.0 / args.num_classes))
        equalise_cls_loss = torch.mean(equalise_cls_loss) * args.num_classes
        equalise_cls_loss = equalise_cls_loss * torch.mean(conf_mask, 0)
    else:
        equalise_cls_loss = None
    return aug_loss, conf_mask, equalise_cls_loss


def compute_aug_loss_double_1(stu_out, tea_out, confidence_thresh, cls_balance=0.001, args=None):
    #need add softmax
    #need to add loss weight
    stu_out = F.softmax(stu_out, dim = 1)
    tea_out = F.softmax(tea_out, dim = 1)
    conf_tea = torch.max(tea_out, 1)[0]
    conf_mask = torch.gt(conf_tea, confidence_thresh).float()

    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss
    aug_loss_weight = d_aug_loss * d_aug_loss * torch.exp(stu_out)
    aug_loss_weight_mean = torch.mean(aug_loss_weight)
    if aug_loss_weight_mean != 0:
        aug_loss_weight = aug_loss_weight * torch.mean(aug_loss) / aug_loss_weight_mean
    aug_loss = aug_loss_weight
    aug_loss = torch.mean(aug_loss, 1) * conf_mask
    if cls_balance > 0.0:
        avg_cls_prob = torch.mean(stu_out, 0)
        equalise_cls_loss = bugged_cls_bal_bce(
                            avg_cls_prob, float(1.0 / args.num_classes))
        equalise_cls_loss = torch.mean(equalise_cls_loss) * args.num_classes
        equalise_cls_loss = equalise_cls_loss * torch.mean(conf_mask, 0)
    else:
        equalise_cls_loss = None
    return aug_loss, conf_mask, equalise_cls_loss



def compute_aug_loss_weight(stu_out, tea_out, confidence_thresh, prob_uk, cls_balance=0.001, args=None):
    #need add softmax
    #need to add loss weight
    stu_out = F.softmax(stu_out, dim = 1)
    tea_out = F.softmax(tea_out, dim = 1)
    conf_tea = torch.max(tea_out, 1)[0]
    conf_mask = torch.gt(conf_tea, confidence_thresh).float()
    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss

    if args.aug_prob_weight == True:
        aug_loss_prob_weight = aug_loss * torch.exp(stu_out)
        aug_loss_prob_weight_mean = torch.mean(aug_loss_prob_weight)
        if aug_loss_prob_weight_mean != 0:
            aug_loss_prob_weight = aug_loss_prob_weight * torch.mean(aug_loss) / aug_loss_prob_weight_mean
        aug_loss = aug_loss_prob_weight

    aug_loss = torch.mean(aug_loss, 1) * conf_mask
    if args.aug_enp_weight == True:
        aug_loss_mean = torch.mean(aug_loss)
        aug_loss_weight = aug_loss * (1 - prob_uk)
        aug_loss_weight_mean = torch.mean(aug_loss_weight)
        if aug_loss_weight_mean != 0:
            aug_loss_weight = aug_loss_weight * (aug_loss_mean / aug_loss_weight_mean)
        aug_loss = aug_loss_weight



    if cls_balance > 0.0:
        avg_cls_prob = torch.mean(stu_out, 0)
        equalise_cls_loss = bugged_cls_bal_bce(
                            avg_cls_prob, float(1.0 / args.num_classes))
        equalise_cls_loss = torch.mean(equalise_cls_loss) * args.num_classes
        equalise_cls_loss = equalise_cls_loss * torch.mean(conf_mask, 0)
    else:
        equalise_cls_loss = None
    return aug_loss, conf_mask, equalise_cls_loss



def new_compute_aug_loss_enp(stu_out, tea_out, confidence_thresh, uk, cls_balance=0.001, args=None):
    stu_out = F.softmax(stu_out, dim=1)
    tea_out = F.softmax(tea_out, dim=1)
    conf_tea = torch.max(tea_out, 1)[0]
    conf_mask = torch.gt(conf_tea, confidence_thresh).float()

    prob_uk = F.sigmoid(uk)
    enp_uk = -1*(prob_uk * torch.log(prob_uk+1e-5)+1e-6).mean(dim=1)
    enp_uk = torch.exp(enp_uk)

    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss

    enp_known =-1*(tea_out * torch.log(tea_out+1e-5)).mean(dim=1)
    enp_known = torch.exp(enp_known).view(-1,1)
    if args.enp_known_weight == True:
        aug_loss_enp_known_weight = aug_loss * enp_known.detach()
        aug_loss_enp_known_mean = torch.mean(aug_loss_enp_known_weight).detach()
        if aug_loss_enp_known_mean >0.1:
            aug_loss_enp_known_weight = aug_loss_enp_known_weight * torch.mean(aug_loss).detach() / (aug_loss_enp_known_mean+1e-5)
        aug_loss = aug_loss_enp_known_weight


    if args.prob_uk:
        weight_prob_uk = torch.exp(1/prob_uk)
        aug_loss_prob_uk = aug_loss * weight_prob_uk.detach()
        aug_loss_prob_uk_mean = torch.mean(aug_loss_prob_uk).detach()
        if aug_loss_prob_uk_mean > 0.1:
            aug_loss_prob_uk = aug_loss_prob_uk * torch.mean(aug_loss).detach() / (aug_loss_prob_uk_mean+1e-5)
        aug_loss = aug_loss_prob_uk
    if cls_balance > 0.0:
        avg_cls_prob = torch.mean(stu_out, 0)
        equalise_cls_loss = bugged_cls_bal_bce(
                            avg_cls_prob, float(1.0 / args.num_classes))
        equalise_cls_loss = torch.mean(equalise_cls_loss) * args.num_classes
        equalise_cls_loss = equalise_cls_loss * torch.mean(conf_mask, 0)
    else:
        equalise_cls_loss = None
    return aug_loss, conf_mask, equalise_cls_loss





def compute_aug_loss_enp(stu_out, tea_out, confidence_thresh, prob_uk, cls_balance=0.001, args=None):
    #need add softmax
    #need to add loss weight
    stu_out = F.softmax(stu_out, dim = 1)
    tea_out = F.softmax(tea_out, dim = 1)
    conf_tea = torch.max(tea_out, 1)[0]
    conf_mask = torch.gt(conf_tea, confidence_thresh).float()

    #prob_uk
    uk_enp = 1 / -1*(prob_uk * torch.log(prob_uk)).mean(dim=1)

    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss
    if args.aug_prob_weight == True:
        aug_loss_prob_weight = aug_loss * torch.exp(stu_out)
        aug_loss_prob_weight_mean = torch.mean(aug_loss_prob_weight)
        if aug_loss_prob_weight_mean != 0:
            aug_loss_prob_weight = aug_loss_prob_weight * torch.mean(aug_loss) / aug_loss_prob_weight_mean
        aug_loss = aug_loss_prob_weight

    aug_loss = torch.mean(aug_loss, 1) * conf_mask

    if args.aug_enp_weight == True:
        aug_loss_mean = torch.mean(aug_loss)
        aug_loss_weight = aug_loss * uk_enp
        aug_loss_weight_mean = torch.mean(aug_loss_weight)
        if aug_loss_weight_mean != 0:
            aug_loss_weight = aug_loss_weight * (aug_loss_mean / aug_loss_weight_mean)
        else:
            aug_loss_weight = aug_loss_weight
        aug_loss = aug_loss_weight
    if cls_balance > 0.0:
        avg_cls_prob = torch.mean(stu_out, 0)
        equalise_cls_loss = bugged_cls_bal_bce(
                            avg_cls_prob, float(1.0 / args.num_classes))
        equalise_cls_loss = torch.mean(equalise_cls_loss) * args.num_classes
        equalise_cls_loss = equalise_cls_loss * torch.mean(conf_mask, 0)
    else:
        equalise_cls_loss = None
    return aug_loss, conf_mask, equalise_cls_loss


def mmdloss(x, y, alpha):
    x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
    y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    K = torch.exp(-1 * alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(-1 * alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(-1 * alpha * (rx.t() + ry - 2*zz))
    B = x.size(0)
    beta = (1./(B*(B)))
    gamma = (2./(B*B))
    loss = beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)
    if loss > 20 or loss< -20:
        loss = 0
    return loss



def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source = source.view(source.size(0), -1)
    target = target.view(target.size(0), -1)
    n_samples = source.size()[0] + target.size()[0]
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwith = fix_sigma
    else:
        bandwith = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwith /= kernel_mul ** (kernel_num // 2)
    bandwith_list = [bandwith * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwith_temp) for bandwith_temp in bandwith_list]
    return sum(kernel_val)

def JAN(source_list, target_list, kernel_muls=[2.0, 2.0, 2.0], kernel_nums=[5, 1, 1], fix_sigma_list=[None, 1.68, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target =  target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss / float(batch_size)


def JAN_unknown(source_list, target_list, source_prob, target_prob, kernel_muls=[2.0, 2.0, 2.0], kernel_nums=[5, 1, 1], fix_sigma_list=[None, 1.68, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    prob = torch.cat([source_prob, target_prob], dim=0).squeeze()
    for i in range(layer_num):
        source = source_list[i]
        target =  target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += joint_kernels[s1, s2] * prob[s1] * prob[s2] + joint_kernels[t1, t2] * prob[t1] * prob[t2]
        loss -= joint_kernels[s1, t2] * prob[s1] * prob[t2] + joint_kernels[s2, t1] * prob[s2] * prob[t1]
    return loss / float(batch_size)


def add_weight(original, weight, confidence_thresh, args):
    if args.add_gan_weight==True:
        original_mean = torch.mean(original)
        original_weight = original * weight
        original_weight_mean = torch.mean(original_weight)
        if original_weight_mean != 0:
            original_weight = original_weight * original_mean / original_weight_mean

        original = original_weight
    conf_mask = torch.gt(weight, confidence_thresh).float()
    original = conf_mask * original
    return torch.mean(original)

from torch.autograd import Function


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)



class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
