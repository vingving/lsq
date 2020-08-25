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

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--bit', default=32, type=int, help='bit-width for lsq quantizer')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name for training')
parser.add_argument('--dataset_size', default=500, type=int, help='dataset size')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
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

args = parser.parse_args()

model_name = 'resnet18_%d_bit_%d'%(args.bit, args.dataset_size)
best_acc = 0  # best test accuracy

# Data
loader = data_loader(args.dataset, args.batch_size, dataset_size=args.dataset_size, path=args.data_root)
testloader = loader.get_test()
if args.dataset is 'distill':
    trainloader = 0
else:
    trainloader = loader.get_train()

if args.dataset == 'cifar10':
    if args.bit == 32:
        net = ResNet18()
        print("Model : Resnet18 32bit (cifar10)")
    else:
        net = Quant_ResNet18(bit=args.bit)
        print("Model : Resnet18 %dbit (cifar10)"%(args.bit))
elif args.dataset == 'imagenet':
    if args.bit == 32:
        net = resnet18(pretrained=True)
        print("Model : Resnet18 32bit (imagenet)")
    else:
        net = quant_resnet18(bit=args.bit)
        print("Model : Resnet18 %dbit (imagenet)" % (args.bit))

net = net.to(device)

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
    # save_path = './checkpoint/resnet18_32_bit/epoch_119.pth'
    save_path = args.home_root + '/.cache/torch/checkpoints/resnet18-5c106cde.pth'  # imagenet
    print('==> Initializing from pre-trained 32bit [%s]' % (save_path))

    checkpoint = torch.load(save_path)

    net_state_dict = net.state_dict()
    net_state_dict.update(checkpoint)
    net.load_state_dict(net_state_dict)

time.sleep(3)

# print("=====================================================")
# for name, parameter in zip(net.state_dict().items(), net.parameters()):
#     print("%s : %d"%(name[0],parameter.requires_grad))

# for name, parameter in zip(net.state_dict().items(), net.parameters()):
#     # print("%s : %d"%(name[0],parameter.requires_grad))
#     # if 'alpha' in name[0] or 'init_state' in name[0]:    ## learn conv & activation quantizaqtion steps
#     if 'quan_a' in name[0]:                                ## learn only activation quantization steps   more like ZeroQ original style
#         parameter.requires_grad = True
#     else:
#         parameter.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

if args.lr_batch_adj:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * math.ceil(len(trainloader) / args.batch_size) ) 
    print ("==> Adjust learing rate according to batch step.")
else:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training
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

    return (train_loss/batch_idx, correct/total)



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
    train_loss, train_acc1 = train(epoch)
    test_loss, test_acc1 = test(epoch)
    print("Epoch time : %s"% format_time(begin_time_epoch-time.time()))
    
    if not args.lr_batch_adj:
        lr_scheduler.step()

    # log the epoch loss
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Acc1', train_acc1, epoch)
    if args.bit != 32:
        writer.add_scalar('Train/layer1[1].conv1.quan_w.alpha', net.layer1[1].conv1.quan_w.alpha, epoch)
        writer.add_scalar('Train/layer2[1].conv1.quan_w.alpha', net.layer2[1].conv1.quan_w.alpha, epoch)

    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Acc1', test_acc1, epoch)

last_time = time.time()
print("Total time : %s"% format_time(last_time-begin_time))
# test_loss, test_acc1 = test(epoch)