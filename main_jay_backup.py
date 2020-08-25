import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import time
import math
import argparse
from models.resnet import *
from models.resnet_quant import *

from models.imagenet_resnet import *
from models.imagenet_resnet_quant import *
from common import data_loader, check_point_manager

# from models.vgg import *
# from models.vgg_quant import *
# from models.vgg_bn import *

from utils import progress_bar, format_time
from torch.utils.tensorboard import SummaryWriter
from distill_data import getDistilData
from pytorchcv.model_provider import get_model as ptcv_get_model
import copy
from lsq import Conv2dLSQ_modify, LinearLSQ_modify

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--bit', default=32, type=int, help='bit-width for lsq quantizer')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name for training')
parser.add_argument('--dataset_size', default=500, type=int, help='dataset size')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--arch', default='resnet18', type=str, help='network architecture')
parser.add_argument('--data_root', default = '/home/com13/data/data', type=str,
                    help='path to dataset')
parser.add_argument('--home_root', default = '/home/com13/', type=str, help='home root')
# ------
parser.add_argument('--init_from', default='./checkpoint', type=str, help='init weights from from checkpoint')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
# ------
parser.add_argument('--lr_batch_adj', default=True, type=bool, help='adjust learning rate according to batch step.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--epochs', default=120, type=int, help='number of training epochs')

parser.add_argument('--distill_size', type=int, default=1000)
parser.add_argument('--distill', default=False, type=bool)

args = parser.parse_args()

model_name = 'resnet18_%d_bit_%d'%(args.bit, args.dataset_size)
best_acc = 0  # best test accuracy

def quantize_model(model, weight_bit):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """

    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d:
        quant_mod = Conv2dLSQ_modify(weight_bit)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = LinearLSQ_modify(weight_bit)
        quant_mod.set_param(model)
        return quant_mod

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, weight_bit))
        return nn.Sequential(*mods)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                # print(attr)
                setattr(q_model, attr, quantize_model(mod, weight_bit))
        return q_model

if args.dataset == 'cifar10':
    if args.bit == 32:
        net = ResNet18()
        print("Model : Resnet18 32bit (cifar10)")
    else:
        net = Quant_ResNet18(bit=args.bit)
        print("Model : Resnet18 %dbit (cifar10)"%(args.bit))
elif args.dataset == 'imagenet':
    if args.distill:
        net2 = resnet18(pretrained=True)
        net = quant_resnet18(bit=args.bit)
        print("Model : Resnet18 %dbit (imagenet)" % (args.bit))
    else:
        net2 = resnet18(pretrained=True)
        net = quant_resnet18(bit=args.bit)
        print("Model : Resnet18 %dbit (imagenet)" % (args.bit))

        # net2 = ptcv_get_model(args.arch, pretrained=True)
        # if args.bit == 32:
        #     net = net2
        #     print("Model : Resnet18 32bit (imagenet)")
        # else:
        #     net = quantize_model(net2, args.bit)
        #     print("Model : Resnet18 %dbit (imagenet)" % (args.bit))

net = net.to(device)
net2 = net2.to(device)

# Data
loader = data_loader(args.dataset, args.batch_size, dataset_size=args.dataset_size, path=args.data_root)
testloader = loader.get_test()
if args.distill :
    if args.distill_size < args.batch_size:
        args.distill_size = args.batch_size
    if args.distill_size % args.batch_size != 0:
        args.distill_size += (args.batch_size - (args.distill_size % args.batch_size))
    print(args.distill_size)
    file_path = os.path.join(args.data_root, 'distill')
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    trainloader = getDistilData(
        net2,
        args.distill_size,
        args.dataset,
        path=file_path,
        # save_mode=True,
        batch_size=args.batch_size)
else:
    trainloader = loader.get_train()

check_manager = check_point_manager(model_name)
save_path, start_epoch = check_manager.load()
if start_epoch > 0:
    # Load checkpoint.
    print('==> Initializing from [%s]'%(save_path))
    checkpoint = torch.load(save_path)

    net_state_dict = net.state_dict()
    net_state_dict.update(checkpoint)
    net.load_state_dict(net_state_dict)

if start_epoch == 0 and args.bit != 32:
    if args.distill:
        save_path = args.home_root + '/.cache/torch/checkpoints/resnet18-5c106cde.pth'  # imagenet
        print('==> Initializing from pre-trained 32bit [%s]' % (save_path))
        checkpoint = torch.load(save_path)
        # checkpoint = net2.state_dict()

        net_state_dict = net.state_dict()
        net_state_dict.update(checkpoint)
        net.load_state_dict(net_state_dict)
    else:
        save_path = args.home_root + '/.cache/torch/checkpoints/resnet18-5c106cde.pth'  # imagenet
        # save_path = args.home_root + '/quantized_model.pth'  # imagenet
        print('==> Initializing from pre-trained 32bit [%s]' % (save_path))
        checkpoint = torch.load(save_path)
        # checkpoint = net2.state_dict()

        net_state_dict = net.state_dict()
        net_state_dict.update(checkpoint)
        net.load_state_dict(net_state_dict)#, strict=False)

time.sleep(3)

if args.distill :
    #### activation only!!!!!!
    # for name, parameter in zip(net.state_dict().items(), net.parameters()):
    #     # print("%s : %d"%(name[0],parameter.requires_grad))
    #     # if 'alpha' in name[0] or 'init_state' in name[0]:    ## learn conv & activation quantizaqtion steps
    #     if 'quan_a' in name[0]  or 'quan_w' in name[0]:         ## learn only activation quantization steps   more like ZeroQ original style
    #         parameter.requires_grad = True
    #     else:
    #         parameter.requires_grad = False

    # print("=====================================================")
    # for name, parameter in zip(net.state_dict().items(), net.parameters()):
    #     print("%s : %d" % (name[0], parameter.requires_grad))

    criterion_distill = nn.BCEWithLogitsLoss()

# print("=====================================================")
# for name, parameter in zip(net.state_dict().items(), net.parameters()):
#     print("%s : %d" % (name[0], parameter.requires_grad))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

if args.lr_batch_adj:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * math.ceil(len(trainloader) / args.batch_size) ) 
    print ("==> Adjust learing rate according to batch step.")
else:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training
def train_distill(epoch):
    print('\nEpoch: %d' % epoch)
    global lr_scheduler, args
    net.train()
    net2.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, inputs in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = F.softmax(net(inputs),1)
        targets = F.softmax(net2(inputs),1)
        loss = criterion_distill(outputs, targets)
        loss.backward()
        optimizer.step()
        if args.lr_batch_adj:
            lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        _, gt = targets.max(1)
        tmp = predicted.eq(gt)
        correct += predicted.eq(gt).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if batch_idx == 0 :
        batch_idx +=1
    return (train_loss/batch_idx, correct/total)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    global lr_scheduler, args
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if args.lr_batch_adj:
            lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    if batch_idx == 0 :
        batch_idx +=1
    return (train_loss / batch_idx, correct / total)



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    if not args.evaluate:
        check_manager.save(net.state_dict(), epoch)
    best_acc = acc

    return (test_loss/batch_idx, correct/total)

if args.evaluate:
    print ("==> Start evaluating ...")
    test(-1)
    exit()


writer = SummaryWriter()

begin_time = time.time()
for epoch in range(start_epoch, args.epochs):
    begin_time_epoch = time.time()
    if args.distill:
        train_loss, train_acc1 = train_distill(epoch)
    else:
        train_loss, train_acc1 = train(epoch)
    test_loss, test_acc1 = test(epoch)
    print("Epoch time : %s"% format_time(begin_time_epoch-time.time()))
    
    if not args.lr_batch_adj:
        lr_scheduler.step()

    # log the epoch loss
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Acc1', train_acc1, epoch)
    state = net.state_dict()
    if args.bit != 32:
        if args.distill:
            writer.add_scalar('Train/layer3[1].conv1.quan_a.alpha', net.layer3[1].conv1.quan_a.alpha, epoch)
            writer.add_scalar('Train/layer2[1].conv1.quan_a.alpha', net.layer2[1].conv1.quan_a.alpha, epoch)
            writer.add_scalar('Train/layer3[1].conv1.quan_w.alpha', net.layer3[1].conv1.quan_w.alpha, epoch)
            writer.add_scalar('Train/layer2[1].conv1.quan_w.alpha', net.layer2[1].conv1.quan_w.alpha, epoch)
        else:
            writer.add_scalar('Train/layer3[1].conv1.quan_a.alpha', state['features.3.0.body.conv1.conv.quan_a.alpha'], epoch)
            writer.add_scalar('Train/layer2[1].conv1.quan_a.alpha', state['features.2.0.body.conv1.conv.quan_a.alpha'], epoch)
            writer.add_scalar('Train/layer3[1].conv1.quan_w.alpha', state['features.3.0.body.conv1.conv.quan_w.alpha'], epoch)
            writer.add_scalar('Train/layer2[1].conv1.quan_w.alpha', state['features.2.0.body.conv1.conv.quan_w.alpha'], epoch)

    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Acc1', test_acc1, epoch)

last_time = time.time()
print("Total time : %s"% format_time(last_time-begin_time))
# test_loss, test_acc1 = test(epoch)