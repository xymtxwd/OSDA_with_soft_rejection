from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from datasets.get_dataset import get_dataset
import models
import utils
import torch.distributions as td
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import pickle
import time
NUM_CLASSES = 5
# Training settings
parser = argparse.ArgumentParser(description='Openset-DA SVHN -> MNIST Example')
parser.add_argument('--task', choices=['s2m', 'u2m', 'm2u'], default='m2u',
                    help='type of task')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
parser.add_argument('--gpu', default='0', type=str, metavar='GPU',
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--seed', type=int, default=1112, metavar='N')

args = parser.parse_args()

start_time = time.time()

import random
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) 
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

source_dataset, target_dataset = get_dataset(args.task)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))
target_test_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, worker_init_fn=np.random.seed(args.seed))
model_f = models.Net_f(task=args.task).cuda()
model_c1 = models.Net_c_cway(task=args.task).cuda()
model_c2 = models.Net_c_cway(task=args.task).cuda()
model_unk = models.disc_digits().cuda()





optimizer_f = torch.optim.Adam(model_f.parameters(), args.lr*1)#, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
optimizer_c = torch.optim.Adam([{"params":model_c1.parameters()},{"params":model_c2.parameters()}], args.lr)#, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
optimizer_unk = torch.optim.Adam(model_unk.parameters(), args.lr)#, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

class LogitBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(LogitBinaryCrossEntropy, self).__init__()
    def forward(self, pred_score, target_score, weights=None):
        loss = F.binary_cross_entropy_with_logits(pred_score, target_score, size_average=True)
        loss = loss * target_score.size(1)
        return loss


criterion_bce = nn.BCELoss()
criterion_cel = nn.CrossEntropyLoss()
criterion_bcewithlogits = LogitBinaryCrossEntropy()

sep_loss_weight = 0
margin = 1
ent_coeff = 1
unk_threshold = 0.5
unk_lb_weight = 0.5
certainty = 0.6
lb_weight = 0.2
unk_aug = False
unk_c = 0.5
upper_threshold = 0.95
lower_threshold = 0.20
tgt_ent_weight = 0.01
if args.task=='s2m':
    tgt_ent_weight = 0.3
step_num = 1
dis_weight=1
use_ent_as_weight = True
with_detach = True
softmax_input_unk = True

best_prec1 = 0
best_pred_y = []
best_gt_y = []
global_step = 0


###### Store all source samples by classes ######

if sep_loss_weight>0:
    class_ind = [[] for x in range(NUM_CLASSES)]
    if args.task=='u2m':
        data_s, target_s = source_dataset.train_data, source_dataset.train_labels
    elif args.task=='s2m':
        data_s, target_s = source_dataset.data, source_dataset.labels
    else:
        data_s, target_s = source_dataset.data, source_dataset.targets
    source_data_by_class = []
    source_data_list_by_class = [[] for x in range(NUM_CLASSES)]

    if args.task=='m2u':
        for i in range(len(data_s)):
            class_ind[target_s[i]].append(i)
            source_data_list_by_class[target_s[i]].append(data_s[i])
        for i in range(NUM_CLASSES):
            source_data_by_class.append(torch.stack(source_data_list_by_class[i]).unsqueeze(1).float().cuda())
    if args.task=='s2m':
        for i in range(len(data_s)):
            class_ind[target_s[i]].append(i)
            source_data_list_by_class[target_s[i]].append(data_s[i])
        for i in range(NUM_CLASSES):
            source_data_by_class.append(torch.tensor(source_data_list_by_class[i]).float().cuda())
    if args.task=='u2m':
        for i in range(len(data_s)):
            class_ind[target_s[i]].append(i)
            source_data_list_by_class[target_s[i]].append(data_s[i])
        for i in range(NUM_CLASSES):
            source_data_by_class.append(torch.tensor(source_data_list_by_class[i]).unsqueeze(1).float().cuda())
    unk_samples = source_data_by_class[-1].float().cuda()



def ent(output):
    return - torch.mean(output * torch.log(output + 1e-6))

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def clip_gradients(myModel, max_norm=0.4):
    norm = nn.utils.clip_grad_norm(myModel.parameters(), max_norm)

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, lr_rampdown_epochs=200):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr *= utils.cosine_rampdown(epoch, lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def pre_train():
    model_f.train()
    model_c1.train()
    model_c2.train()
    model_unk.train()
    pred_y = []
    correct = 0
    for batch_idx, (data_s, target_s) in enumerate(source_loader):

        data_s, target_s = data_s.cuda(), target_s.cuda(non_blocking=True)
        batch_size_s = len(target_s)

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_unk.zero_grad()
        feature_s = model_f(data_s)
        output_s1 = model_c1(feature_s)
        output_s2 = model_c2(feature_s)

        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()<NUM_CLASSES-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        loss_s = criterion_cel(F.softmax(output_s1).index_select(0, ind), target_s.index_select(0, ind))
        loss_s += criterion_cel(F.softmax(output_s2).index_select(0, ind), target_s.index_select(0, ind))

        output_s_prob_sft1 = F.softmax(output_s1, dim=1)
        output_s_prob_sft2 = F.softmax(output_s2, dim=1)
        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()==NUM_CLASSES-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        loss_s -= td.Categorical(probs=output_s_prob_sft1.index_select(0, ind)).entropy().mean()*ent_coeff
        loss_s -= td.Categorical(probs=output_s_prob_sft2.index_select(0, ind)).entropy().mean()*ent_coeff

        loss = loss_s
        loss.backward()

        optimizer_f.step()
        optimizer_c.step()
        #optimizer_unk.step()

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_unk.zero_grad()



        output = output_s1 + output_s2
        pred = output.max(1, keepdim=True)[1]
        for i in range(len(pred)):
            pred_y.append(pred[i].item())
        correct += pred.eq(target_s.view_as(pred)).sum().item()
    #print('source acc: '+str(100* correct / len(source_loader.dataset)))

def pre_train_unk():
    model_unk.train()
    pred_y = []
    correct = 0
    for batch_idx, (data_s, target_s) in enumerate(source_loader):

        data_s, target_s = data_s.cuda(), target_s.cuda(non_blocking=True)
        batch_size_s = len(target_s)

        optimizer_unk.zero_grad()
        feature_s = model_f(data_s)
        output_s1 = model_c1(feature_s)
        output_s2 = model_c2(feature_s)

        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()<NUM_CLASSES-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        if softmax_input_unk:
            unk_loss = criterion_bce(model_unk(F.softmax(output_s1, dim=1).index_select(0, ind)), torch.zeros_like(ind).float().cuda())
            unk_loss += criterion_bce(model_unk(F.softmax(output_s2, dim=1).index_select(0, ind)), torch.zeros_like(ind).float().cuda())
        else:
            unk_loss = criterion_bce(model_unk(output_s1.index_select(0, ind)), torch.zeros_like(ind).float().cuda())
            unk_loss += criterion_bce(model_unk(output_s2.index_select(0, ind)), torch.zeros_like(ind).float().cuda())

        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()==NUM_CLASSES-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()

        if softmax_input_unk:
            unk_loss += criterion_bce(model_unk(F.softmax(output_s1, dim=1).index_select(0, ind)), torch.ones_like(ind).float().cuda())
            unk_loss += criterion_bce(model_unk(F.softmax(output_s2, dim=1).index_select(0, ind)), torch.ones_like(ind).float().cuda())
        else:
            unk_loss += criterion_bce(model_unk(output_s1.index_select(0, ind)), torch.ones_like(ind).float().cuda())
            unk_loss += criterion_bce(model_unk(output_s2.index_select(0, ind)), torch.ones_like(ind).float().cuda())


        loss = unk_loss
        loss.backward()
        #print(loss.item())
        optimizer_unk.step()
        optimizer_f.step()
        optimizer_c.step()

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_unk.zero_grad()

        output = output_s1 + output_s2
        pred = output.max(1, keepdim=True)[1]
        for i in range(len(pred)):
            pred_y.append(pred[i].item())
        correct += pred.eq(target_s.view_as(pred)).sum().item()
    #print('source acc: '+str(100* correct / len(source_loader.dataset)))

def train(epoch):
    model_f.train()
    model_c1.train()
    model_c2.train()
    model_unk.train()
    global global_step
    for batch_idx, (batch_s, batch_t) in enumerate(zip(source_loader, target_loader)):
        adjust_learning_rate(optimizer_f, epoch, batch_idx, len(source_loader))
        adjust_learning_rate(optimizer_c, epoch, batch_idx, len(source_loader))
        data_s, target_s = batch_s
        data_t, target_t = batch_t
        data_s, target_s = data_s.cuda(), target_s.cuda(non_blocking=True)
        data_t, target_t = data_t.cuda(), target_t.cuda(non_blocking=True)
        batch_size_s = len(target_s)
        batch_size_t = len(target_t)

###### STEP 1 ######

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_unk.zero_grad()
        feature_s = model_f(data_s)
        output_s1 = model_c1(feature_s)
        output_s2 = model_c2(feature_s)


        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()<NUM_CLASSES-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        loss_s = criterion_cel(output_s1.index_select(0, ind), target_s.index_select(0, ind))
        loss_s += criterion_cel(output_s2.index_select(0, ind), target_s.index_select(0, ind))
        unk_loss = torch.tensor(0).float().cuda()
        if with_detach:
            if softmax_input_unk:
                unk_loss += criterion_bce(model_unk(F.softmax(output_s1, dim=1).index_select(0, ind).detach()), torch.zeros_like(ind).float().cuda())
                unk_loss += criterion_bce(model_unk(F.softmax(output_s2, dim=1).index_select(0, ind).detach()), torch.zeros_like(ind).float().cuda())
            else:
                unk_loss += criterion_bce(model_unk(output_s1.index_select(0, ind).detach()), torch.zeros_like(ind).float().cuda())
                unk_loss += criterion_bce(model_unk(output_s2.index_select(0, ind).detach()), torch.zeros_like(ind).float().cuda())
        else:
            if softmax_input_unk:
                unk_loss += criterion_bce(model_unk(F.softmax(output_s1).index_select(0, ind)), torch.zeros(len(ind)).float().cuda())
                unk_loss += criterion_bce(model_unk(F.softmax(output_s2).index_select(0, ind)), torch.zeros(len(ind)).float().cuda())
            else:
                unk_loss += criterion_bce(model_unk(output_s1.index_select(0, ind)), torch.zeros(len(ind)).float().cuda())
                unk_loss += criterion_bce(model_unk(output_s2.index_select(0, ind)), torch.zeros(len(ind)).float().cuda())


        feature_t = model_f(data_t)
        output_t1 = model_c1(feature_t)
        output_t2 = model_c2(feature_t)


        output_s_prob_sft1 = F.softmax(output_s1, dim=1)
        output_s_prob_sft2 = F.softmax(output_s2, dim=1)

        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()==NUM_CLASSES-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        loss_s -= td.Categorical(probs=output_s_prob_sft1.index_select(0, ind)).entropy().mean()*ent_coeff
        loss_s -= td.Categorical(probs=output_s_prob_sft2.index_select(0, ind)).entropy().mean()*ent_coeff

        if with_detach:
            if softmax_input_unk:
                unk_loss += criterion_bce(model_unk(F.softmax(output_s1, dim=1).index_select(0, ind).detach()), torch.ones_like(ind).float().cuda())
                unk_loss += criterion_bce(model_unk(F.softmax(output_s2, dim=1).index_select(0, ind).detach()), torch.ones_like(ind).float().cuda())
            else:
                unk_loss += criterion_bce(model_unk(output_s1.index_select(0, ind).detach()), torch.ones_like(ind).float().cuda())
                unk_loss += criterion_bce(model_unk(output_s2.index_select(0, ind).detach()), torch.ones_like(ind).float().cuda())
        else:
            if softmax_input_unk:
                unk_loss += criterion_bce(model_unk(F.softmax(output_s1).index_select(0, ind)), torch.ones(len(ind)).float().cuda())
                unk_loss += criterion_bce(model_unk(F.softmax(output_s2).index_select(0, ind)), torch.ones(len(ind)).float().cuda())
            else:
                unk_loss += criterion_bce(model_unk(output_s1.index_select(0, ind)), torch.ones(len(ind)).float().cuda())
                unk_loss += criterion_bce(model_unk(output_s2.index_select(0, ind)), torch.ones(len(ind)).float().cuda())


        #mmd_loss = compute_mmd(feature_s.mean(dim=0).unsqueeze(dim=0), feature_t.mean(dim=0).unsqueeze(dim=0))

        lb = criterion_bcewithlogits(F.softmax(output_t1, dim=1).mean(dim=0).unsqueeze(dim=0), torch.tensor([1/(NUM_CLASSES-1)]*(NUM_CLASSES-1) ).unsqueeze(dim=0).cuda())
        lb += criterion_bcewithlogits(F.softmax(output_t2, dim=1).mean(dim=0).unsqueeze(dim=0), torch.tensor([1/(NUM_CLASSES-1)]*(NUM_CLASSES-1) ).unsqueeze(dim=0).cuda())

        if softmax_input_unk:
            lb += unk_lb_weight*torch.abs(model_unk(F.softmax(output_t1, dim=1)).mean() - torch.tensor(0.5).cuda())
            lb += unk_lb_weight*torch.abs(model_unk(F.softmax(output_t2, dim=1)).mean() - torch.tensor(0.5).cuda())
        else:
            lb += unk_lb_weight*torch.abs(model_unk(output_t1).mean() - torch.tensor(0.5).cuda())
            lb += unk_lb_weight*torch.abs(model_unk(output_t2).mean() - torch.tensor(0.5).cuda())


        #output = output_t1 + output_t2
        #pred = output.max(1, keepdim=True)[1]
        #pred.requires_grad = False
        #pred.detach()
        #print(pred)

        sample_ent1 = td.Categorical(probs=F.softmax(output_t1)).entropy()
        sample_ent2 = td.Categorical(probs=F.softmax(output_t2)).entropy()
        sample_weight = torch.exp(-(sample_ent1 + sample_ent2)*0.5)

        centroid_loss = torch.tensor(0).cuda()
        if False:#epoch > 3: 
            len_ = len(pred)
        else:
            len_ = 0
        for i in range(0, 0):#len(pred)):
            tmp_ = data_t[i].clone()
            tmp_ = tmp_.unsqueeze_(0)
            index_ = pred[i].item()
            rd_ind = random.sample(range(0, len(source_data_by_class[index_])), np.min([32,len(source_data_by_class[index_])]) )
            rd_ind = torch.tensor(rd_ind).cuda()
            if index_!=NUM_CLASSES-1:
                centroid_loss = centroid_loss + sample_weight[i]*torch.norm(model_f(source_data_by_class[index_].index_select(0, rd_ind)).mean(dim=0) - model_f(tmp_),2)
        #print(centroid_loss.item())

###### unk separate loss

        unk_sep_loss = torch.tensor(0).float().cuda()
        if sep_loss_weight>0:
            ind_list = []
            for i in range(batch_size_s):
                if target_s[i].item()<NUM_CLASSES-1:
                    ind_list.append(i)
            ind = torch.tensor(ind_list).cuda()
            rd_ind = random.sample(range(0, len(unk_samples)), np.min([32,len(unk_samples)]) )
            rd_ind = torch.tensor(rd_ind).cuda()
            temp_unk = unk_samples.index_select(0, rd_ind)
            unk_feat = model_f(unk_samples.index_select(0, rd_ind))
            s_feat = feature_s.index_select(0, ind)
            s_cent_feat = []
    
            for i in range(NUM_CLASSES-1):
                rd_ind_cent = random.sample(range(0, len(source_data_by_class[i])), np.min([32,len(source_data_by_class[i])]) )
                rd_ind_cent = torch.tensor(rd_ind_cent).cuda()
                s_cent_feat.append(model_f(source_data_by_class[i].index_select(0, rd_ind_cent)).mean(dim=0))
    
            for i in range(len(ind)):
                for j in range(len(rd_ind)):
                    tmp_loss = F.relu(torch.tensor(margin).float().cuda() + torch.norm(s_feat[i] - s_cent_feat[target_s.index_select(0, ind)[i].item()], 2) - torch.norm(s_feat[i] - unk_feat[j], 2))/len(ind)
                    unk_sep_loss += tmp_loss
                    #print(tmp_loss)
                
        #print(loss_s.item())
        #print(unk_loss.item())
        #print(0.0005*centroid_loss.item())
        #print('-------------')


        loss_target_ent = td.Categorical(probs=(F.softmax(output_t1)+F.softmax(output_t2))/2).entropy()

        if use_ent_as_weight:
            loss_target_ent= (loss_target_ent * sample_weight / sample_weight.sum()).sum()
        else:
            loss_target_ent = td.Categorical(probs=(output_t1+output_t2)/2).entropy().mean()

        loss = 1*loss_s + 1* unk_loss + lb_weight*lb + tgt_ent_weight*loss_target_ent + sep_loss_weight*unk_sep_loss

        loss.backward()
        clip_gradients(model_f)
        clip_gradients(model_c1)
        clip_gradients(model_c2)
        clip_gradients(model_unk)

        optimizer_f.step()
        optimizer_c.step()
        optimizer_unk.step()

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_unk.zero_grad()


        if False:
            feature_s = model_f(data_s)
            output_s1 = model_c1(feature_s)
            output_s2 = model_c2(feature_s)

            ind_list = []
            for i in range(batch_size_s):
                if target_s[i].item()<NUM_CLASSES-1:
                    ind_list.append(i)
            ind = torch.tensor(ind_list).cuda()

            unk_loss = criterion_bce(model_unk(output_s1.index_select(0, ind)), torch.zeros(len(ind)).float().cuda())
            unk_loss += criterion_bce(model_unk(output_s2.index_select(0, ind)), torch.zeros(len(ind)).float().cuda())

            ind_list = []
            for i in range(batch_size_s):
                if target_s[i].item()==NUM_CLASSES-1:
                    ind_list.append(i)
            ind = torch.tensor(ind_list).cuda()
            unk_loss += criterion_bce(model_unk(output_s1.index_select(0, ind)), torch.ones(len(ind)).float().cuda())
            unk_loss += criterion_bce(model_unk(output_s2.index_select(0, ind)), torch.ones(len(ind)).float().cuda())

            loss = unk_loss
            loss.backward()
            optimizer_unk.step()
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            optimizer_unk.zero_grad()


    ### UNK/K augmentation
        if unk_aug and epoch > 5:
            ind_k = []
            ind_unk = []
            din_t1 = F.softmax(output_t1).detach()
            din_t2 = F.softmax(output_t2).detach()
            #print((F.softmax(output_t1)+F.softmax(output_t2))/2)
            max_sim = torch.max((F.softmax(output_t1)+F.softmax(output_t2))/2, dim=1)[0]
            #print(max_sim)
            for i in range(0, len(output_t1)):
                if max_sim[i].item()>upper_threshold:
                    ind_k.append(i)
                if max_sim[i].item()<lower_threshold:
                    ind_unk.append(i)
            ind_k_ = torch.tensor(ind_k).cuda()
            ind_unk_ = torch.tensor(ind_unk).cuda()
            print(len(ind_k_))
            print(len(ind_unk_))
            if len(ind_k_)>1:
                unk_loss += unk_c*criterion_bce(model_unk(din_t1.index_select(0, ind_k_)), torch.zeros_like(ind_k_).float().cuda())
                unk_loss += unk_c*criterion_bce(model_unk(din_t2.index_select(0, ind_k_)), torch.zeros_like(ind_k_).float().cuda())

            elif len(ind_k_)==1:
                unk_loss += unk_c*criterion_bce(model_unk(din_t1.index_select(0, ind_k_).clone().unsqueeze_(0)), torch.zeros_like(ind_k_).float().cuda())
                unk_loss += unk_c*criterion_bce(model_unk(din_t2.index_select(0, ind_k_).clone().unsqueeze_(0)), torch.zeros_like(ind_k_).float().cuda())

    
            if len(ind_unk_)>1:
                unk_loss += unk_c*criterion_bce(model_unk(din_t1.index_select(0, ind_unk_)), torch.ones_like(ind_unk_).float().cuda())
                unk_loss += unk_c*criterion_bce(model_unk(din_t2.index_select(0, ind_unk_)), torch.ones_like(ind_unk_).float().cuda())

            elif len(ind_unk_)==1:
                unk_loss += unk_c*criterion_bce(model_unk(din_t1.index_select(0, ind_unk_).clone().unsqueeze_(0)), torch.ones_like(ind_unk_).float().cuda())
                unk_loss += unk_c*criterion_bce(model_unk(din_t2.index_select(0, ind_unk_).clone().unsqueeze_(0)), torch.ones_like(ind_unk_).float().cuda())


###### STEP 2 ######


        feature_s = model_f(data_s)
        output_s1 = model_c1(feature_s)
        output_s2 = model_c2(feature_s)

        feature_t = model_f(data_t)
        output_t1 = model_c1(feature_t)
        output_t2 = model_c2(feature_t)


        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()<NUM_CLASSES-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        loss_s = criterion_cel(F.softmax(output_s1).index_select(0, ind), target_s.index_select(0, ind))
        loss_s += criterion_cel(F.softmax(output_s2).index_select(0, ind), target_s.index_select(0, ind))
        #unk_loss = criterion_bce(model_unk(feature_s.index_select(0, ind)), torch.zeros(len(ind)).float().cuda())


        output_s_prob_sft1 = F.softmax(output_s1, dim=1)
        output_s_prob_sft2 = F.softmax(output_s2, dim=1)

        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()==NUM_CLASSES-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        loss_s -= td.Categorical(probs=output_s_prob_sft1.index_select(0, ind)).entropy().mean()*ent_coeff
        loss_s -= td.Categorical(probs=output_s_prob_sft2.index_select(0, ind)).entropy().mean()*ent_coeff


        if use_ent_as_weight:
            sample_ent1 = td.Categorical(probs=F.softmax(output_t1)).entropy()
            sample_ent2 = td.Categorical(probs=F.softmax(output_t2)).entropy()
            sample_weight = torch.exp(-(sample_ent1 + sample_ent2)*0.5)
            loss_dis = torch.mean(torch.abs(F.softmax(output_t1)-F.softmax(output_t2)), dim=1)
            loss_dis = (loss_dis * sample_weight / sample_weight.sum()).sum()
        else:
            loss_dis = discrepancy(F.softmax(output_t1, dim=1), F.softmax(output_t2, dim=1))



        loss = loss_s - dis_weight*loss_dis

        loss.backward()

        optimizer_c.step()

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_unk.zero_grad()


###### STEP 3 ######

        for i in range(step_num):
            feat_t = model_f(data_t)
            output_t1 = model_c1(feat_t)
            output_t2 = model_c2(feat_t)

            if use_ent_as_weight:
                sample_ent1 = td.Categorical(probs=F.softmax(output_t1)).entropy()
                sample_ent2 = td.Categorical(probs=F.softmax(output_t2)).entropy()
                sample_weight = torch.exp(-(sample_ent1 + sample_ent2)*0.5)
                loss_dis = torch.mean(torch.abs(F.softmax(output_t1)-F.softmax(output_t2)), dim=1)
                loss_dis = (loss_dis * sample_weight / sample_weight.sum()).sum()
            else:
                loss_dis = discrepancy(output_t1, output_t2)


            loss_dis.backward()
#            print(loss_dis.item())
#            print('-------------')

            clip_gradients(model_f)

            optimizer_f.step()
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            optimizer_unk.zero_grad()




        global_step += 1
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * args.batch_size, len(source_loader.dataset),
                100. * batch_idx / len(source_loader), loss.item()))

def test(epoch):
    global best_prec1
    model_f.eval()
    model_c1.eval()
    model_c2.eval()
    model_unk.eval()
    loss = 0
    pred_y = []
    true_y = []
    unk_pred_y = []
    unk_pred_count = 0

    correct = 0
    ema_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(target_test_loader):
            data, target = data.cuda(), target.cuda(non_blocking=True)
            features = model_f(data)
            output1 = model_c1(features)
            output2 = model_c2(features)

            if softmax_input_unk:
                output_unk1 = model_unk(F.softmax(output1))
                output_unk2 = model_unk(F.softmax(output2))
            else:
                output_unk1 = model_unk(output1)
                output_unk2 = model_unk(output2)
            output_unk = (output_unk1+output_unk2)/2

            output = output1 + output2

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            for i in range(len(pred)):
                if output_unk[i].item()>unk_threshold and torch.max(F.softmax(output), dim=1)[0][i].item()<certainty:
                    pred_y.append(NUM_CLASSES-1)
                else:
                    pred_y.append(pred[i].item())
                true_y.append(target[i].item())

            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(target_loader.dataset)

    avg_acc = utils.cal_acc(true_y, pred_y, NUM_CLASSES)
    prec1 = avg_acc

    if epoch % 1 == 0:
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model_f.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer_f.state_dict(),
        }, is_best)
        if is_best:
            global best_gt_y
            global best_pred_y
            best_gt_y = true_y
            best_pred_y = pred_y
            with open(str(args.seed) + args.task + '_f.pkl', 'wb') as output_file:
                pickle.dump(model_f, output_file)
            with open(str(args.seed) + args.task + '_c1.pkl', 'wb') as output_file:
                pickle.dump(model_c1, output_file)
            with open(str(args.seed) + args.task + '_c2.pkl', 'wb') as output_file:
                pickle.dump(model_c2, output_file)
            with open(str(args.seed) + args.task + '_unk.pkl', 'wb') as output_file:
                pickle.dump(model_unk, output_file)

            #model_f.save_state_dict(str(args.seed) + args.task + '_f.pt')
            #model_c1.save_state_dict(str(args.seed) + args.task + '_c1.pt')
            #model_c2.save_state_dict(str(args.seed) + args.task + '_c2.pt')
            #model_unk.save_state_dict(str(args.seed) + args.task + '_unk.pt')

    print('Best test accuracy: ' + str(best_prec1))


for epoch in range(1, 5):
    pre_train()
for epoch in range(1, 10):
    pre_train_unk()
if args.task=='u2m':
    for epoch in range(1, 15):
        pre_train()
        pre_train_unk()
try:
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
    print ("------Best Result-------")
    utils.cal_acc(best_gt_y, best_pred_y, NUM_CLASSES)
    print("--- %s seconds ---" % (time.time() - start_time))
except KeyboardInterrupt:
    print ("------Best Result-------")
    utils.cal_acc(best_gt_y, best_pred_y, NUM_CLASSES)