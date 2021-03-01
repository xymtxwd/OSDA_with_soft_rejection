from __future__ import print_function
import argparse
from utils.utils import *
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data_loader.get_loader import get_loader
import numpy as np
import os
from models.basenet import disc
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Openset DA')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--net', type=str, default='vgg', metavar='B',
                    help='which network alex,vgg,res?')
parser.add_argument('--save', action='store_true', default=False,
                    help='save model or not')
parser.add_argument('--save_path', type=str, default='checkpoint/checkpoint', metavar='B',
                    help='checkpoint path')
parser.add_argument('--source_path', type=str, default='./utils/source_list__.txt', metavar='B',
                    help='checkpoint path')
parser.add_argument('--target_path', type=str, default='./utils/target_list__.txt', metavar='B',
                    help='checkpoint path')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--unit_size', type=int, default=1000, metavar='N',
                    help='unit size of fully connected layer')
parser.add_argument('--update_lower', action='store_true', default=True,
                    help='update lower layer or not')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable cuda')
parser.add_argument('--opt', default='sgd')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

source_data = args.source_path
target_data = args.target_path
evaluation_data = args.target_path
batch_size = args.batch_size
data_transforms = {
    source_data: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    target_data: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    evaluation_data: transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def clip_gradients(myModel, max_norm=0.4):
    norm = nn.utils.clip_grad_norm(myModel.parameters(), max_norm)
class LogitBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(LogitBinaryCrossEntropy, self).__init__()
    def forward(self, pred_score, target_score, weights=None):
        loss = F.binary_cross_entropy_with_logits(pred_score, target_score, size_average=True)
        loss = loss * target_score.size(1)
        return loss



use_gpu = torch.cuda.is_available()

import random
torch.backends.cudnn.benchmark = True
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) 
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

train_loader, test_loader = get_loader(source_data, target_data, evaluation_data,
                                       data_transforms, batch_size=args.batch_size)
dataset_train = train_loader.load_data()
dataset_test = test_loader

num_class = 7
class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]
model_f, model_c1, model_c2 = get_model(args.net, num_class=num_class-1, unit_size=args.unit_size)
model_unk = disc().cuda()

if args.cuda:
    model_f.cuda()
    model_c1.cuda()
    model_c2.cuda()

criterion_bce = nn.BCELoss()
criterion_cel = nn.CrossEntropyLoss()
criterion_bcewithlogits = LogitBinaryCrossEntropy()

sep_loss_weight = 0
margin = 1
ent_coeff = 1
unk_threshold = 0.5
unk_lb_weight = 0.5
certainty = 1
lb_weight = 0.2
unk_aug = False
unk_c = 0.5
upper_threshold = 0.95
lower_threshold = 0.20
tgt_ent_weight = 0.01
step_num = 4
dis_weight=1
use_ent_as_weight = True
with_detach = True
softmax_input_unk = True

best_prec1 = 0
best_pred_y = []
best_gt_y = []
global_step = 0


#optimizer_g = opt.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005,nesterov=True)
#optimizer_c = opt.SGD(list(C.parameters()), momentum=0.9, lr=lr,
#                          weight_decay=0.0005, nesterov=True)

param = list(list(model_f.linear1.parameters()) + list(model_f.linear2.parameters()) + list(model_f.bn1.parameters()) + list(model_f.bn2.parameters()))
if args.opt=='adam':
    optimizer_f = torch.optim.Adam(model_f.parameters(), 0.0001)#, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_c = torch.optim.Adam([{"params":model_c1.parameters()},{"params":model_c2.parameters()}], 0.0001)#, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_unk = torch.optim.Adam(model_unk.parameters(), 0.0001)#, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
if args.opt=='sgd':
    optimizer_f = torch.optim.SGD(param, 0.0001, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_c = torch.optim.SGD([{"params":model_c1.parameters()},{"params":model_c2.parameters()}], 0.0001, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_unk = torch.optim.SGD(model_unk.parameters(), 0.0001, momentum=0.9, weight_decay=0.0005, nesterov=True)

print(args.save_path)


def pre_train():
    model_f.train()
    model_c1.train()
    model_c2.train()
    model_unk.train()
    pred_y = []
    correct = 0
    for batch_idx, data in enumerate(dataset_train):
        if batch_idx * batch_size > 30000:
            break
        data_s, target_s = data['S'].cuda(), data['S_label'].cuda(non_blocking=True).long()

        batch_size_s = len(target_s)

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_unk.zero_grad()
        feature_s = model_f(data_s)
        output_s1 = model_c1(feature_s)
        output_s2 = model_c2(feature_s)

        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()<num_class-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        loss_s = criterion_cel(F.softmax(output_s1).index_select(0, ind), target_s.index_select(0, ind))
        loss_s += criterion_cel(F.softmax(output_s2).index_select(0, ind), target_s.index_select(0, ind))

        output_s_prob_sft1 = F.softmax(output_s1, dim=1)
        output_s_prob_sft2 = F.softmax(output_s2, dim=1)
        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()==num_class-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        if len(ind_list)!=0:
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
    print('source acc: '+str(100* correct / len(pred_y)))



def train(epoch):
    model_f.train()
    model_c1.train()
    model_c2.train()
    model_unk.train()
    global global_step
    for batch_idx, data in enumerate(dataset_train):
        data_s, target_s = data['S'].cuda(), data['S_label'].cuda(non_blocking=True).long()
        data_t = data['T'].cuda()
        batch_size_s = len(data_s)
        batch_size_t = len(data_t)
        if batch_size_t!=batch_size_s: break
        if batch_idx * batch_size > 30000:
            break
###### STEP 1 ######

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_unk.zero_grad()
        feature_s = model_f(data_s)
        output_s1 = model_c1(feature_s)
        output_s2 = model_c2(feature_s)


        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()<num_class-1:
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
        output_t1 = model_c1(feature_t)#, constant = constant, adaption = True)
        output_t2 = model_c2(feature_t)#, constant = constant, adaption = True)



        output_s_prob_sft1 = F.softmax(output_s1, dim=1)
        output_s_prob_sft2 = F.softmax(output_s2, dim=1)


        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()==num_class-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        if len(ind_list)!=0:
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

        lb = criterion_bcewithlogits(F.softmax(output_t1, dim=1).mean(dim=0).unsqueeze(dim=0), torch.tensor([1/(num_class-1)]*(num_class-1) ).unsqueeze(dim=0).cuda())
        lb += criterion_bcewithlogits(F.softmax(output_t2, dim=1).mean(dim=0).unsqueeze(dim=0), torch.tensor([1/(num_class-1)]*(num_class-1) ).unsqueeze(dim=0).cuda())

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

        loss_target_ent = td.Categorical(probs=(F.softmax(output_t1)+F.softmax(output_t2))/2).entropy()

        if use_ent_as_weight:
            loss_target_ent= (loss_target_ent * sample_weight / sample_weight.sum()).sum()
        else:
            loss_target_ent = td.Categorical(probs=(output_t1+output_t2)/2).entropy().mean()


        loss = 1*loss_s + 1* unk_loss + lb_weight*lb + tgt_ent_weight*loss_target_ent


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

###### STEP 2 ######


        feature_s = model_f(data_s)
        output_s1 = model_c1(feature_s)#, constant = constant, adaption = True)
        output_s2 = model_c2(feature_s)#, constant = constant, adaption = True)

        feature_t = model_f(data_t)
        output_t1 = model_c1(feature_t)#, constant = constant, adaption = True)
        output_t2 = model_c2(feature_t)#, constant = constant, adaption = True)


        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()<num_class-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()
        loss_s = criterion_cel(F.softmax(output_s1).index_select(0, ind), target_s.index_select(0, ind))
        loss_s += criterion_cel(F.softmax(output_s2).index_select(0, ind), target_s.index_select(0, ind))
        #unk_loss = criterion_bce(model_unk(feature_s.index_select(0, ind)), torch.zeros(len(ind)).float().cuda())


        output_s_prob_sft1 = F.softmax(output_s1, dim=1)
        output_s_prob_sft2 = F.softmax(output_s2, dim=1)

        ind_list = []
        for i in range(batch_size_s):
            if target_s[i].item()==num_class-1:
                ind_list.append(i)
        ind = torch.tensor(ind_list).cuda()

        if len(ind_list)!=0:
            loss_s -= td.Categorical(probs=output_s_prob_sft1.index_select(0, ind)).entropy().mean()*ent_coeff
            loss_s -= td.Categorical(probs=output_s_prob_sft2.index_select(0, ind)).entropy().mean()*ent_coeff


        if use_ent_as_weight:
            sample_ent1 = td.Categorical(probs=F.softmax(output_t1)).entropy()
            sample_ent2 = td.Categorical(probs=F.softmax(output_t2)).entropy()
            sample_weight = torch.exp(-(sample_ent1 + sample_ent2)*0.5)
            loss_dis = torch.mean(torch.abs(F.softmax(output_t1)-F.softmax(output_t2)), dim=1)
            loss_dis = (loss_dis * sample_weight / sample_weight.sum()).sum()
        else:
            loss_dis = torch.mean(torch.abs(F.softmax(output_t1)-F.softmax(output_t2)))


        loss = loss_s - dis_weight*loss_dis#+ 0.0001*centroid_loss #+unk_loss#+ 0.5*mmd_loss   #+ entropy_loss*0.01

        loss.backward()

        optimizer_c.step()
        #optimizer_unk.step()

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
                loss_dis = torch.mean(torch.abs(F.softmax(output_t1)-F.softmax(output_t2)))


            loss_dis.backward()

            clip_gradients(model_f)
            optimizer_f.step()
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            optimizer_unk.zero_grad()
        if epoch > 0 and batch_idx % 1000 == 0:
            test()
            model_f.train()
            model_c1.train()
            model_c2.train()
            model_unk.train()


def test():
    model_f.eval()
    model_c1.eval()
    model_c2.eval()
    model_unk.eval()
    correct = 0
    size = 0

    per_class_num = np.zeros((num_class))
    per_class_correct = np.zeros((num_class)).astype(np.float32)
    for batch_idx, data in enumerate(dataset_test):
        if args.cuda:
            img_t, label_t = data[0], data[1]
            data, target = Variable(img_t.cuda(), volatile=True), Variable(label_t.cuda().long(), volatile=True)

        features = model_f(data)
        output1 = model_c1(features)
        output2 = model_c2(features)
        pred_y = []
        if softmax_input_unk:
            output_unk1 = model_unk(F.softmax(output1))
            output_unk2 = model_unk(F.softmax(output2))
        else:
            output_unk1 = model_unk(output1)
            output_unk2 = model_unk(output2)
        output_unk = (output_unk1+output_unk2)/2
        output = output1 + output2

        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if True:
            for i in range(len(pred)):
                if output_unk[i].item()>unk_threshold and torch.max(F.softmax(output), dim=1)[0][i].item()<certainty:
                    pred_y.append(num_class-1)
                else:
                    pred_y.append(pred[i].item())

        pred_y = np.array(pred_y)
        for t in range(num_class):
            t_ind = np.where(target.data.cpu().numpy() == t)
            correct_ind = np.where(pred_y[t_ind[0]] == t)
            per_class_correct[t] += float(len(correct_ind[0]))
            per_class_num[t] += float(len(t_ind[0]))
    per_class_acc = per_class_correct / per_class_num

    #print(
    #    '\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)  ({:.4f}%)\n'.format(
    #        correct, size,
    #        100. * correct / size, float(per_class_acc.mean())))
    for ind, category in enumerate(class_list):
        print('%s:%s' % (category, per_class_acc[ind]))

    print('Average accuracy:')
    print(float(per_class_acc.mean()))
for i in range(2):
    pre_train()
for i in range(50):
    train(i)
    test()
#train(args.epochs + 1)
