import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import math
import argparse
from torchsummary import summary
os.environ['CUDA_VISIBLE_DEVICES']='0'
# from models.vgg import *
# from models.vgg_quant import *
# from models.vgg_bn import *



from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--bit', default=32, type=int, help='bit-width for lsq quantizer')

parser.add_argument('--dataset', default='imagenet', type=str,
                    help='dataset name for training')
parser.add_argument('--data_root', default = '/home/com13/data', type=str,
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='network architecture')
# ------
parser.add_argument('--init_from', type=str,
                    help='init weights from from checkpoint')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# ------
parser.add_argument('--lr_batch_adj', action='store_true',
                    help='adjust learning rate according to batch step.')


parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--epochs', default=120, type=int, help='number of training epochs')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
writer = SummaryWriter()

# Data
print('==> Preparing data..')
if args.dataset == "cifar10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

elif args.dataset == "imagenet":
    traindir = os.path.join(args.data_root, 'val')
    valdir = os.path.join(args.data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

# Model
print('==> Building model..')
# net = vgg16_bn()
if args.dataset == "cifar10":
    if args.arch == "resnet18":
        if args.bit == 32:
            pass
        else:
            from models.resnet import *
            from models.resnet_quant import *
            net = Quant_ResNet18(bit=args.bit)
    elif args.arch == "resnext29_32x4d":
        from models.cifar_resnext import *
        net = ResNeXt29_32x4d(bit=args.bit)


elif args.dataset == "imagenet":
    if args.arch == "resnet18":
        if args.bit == 32:
            from models.imagenet_resnet import *
            net = resnet18(pretrained=True)
        else:
            from models.imagenet_resnet_quant import *
            net = quant_resnet18(bit=args.bit)
    else:
        pass

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()

net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.dataset == "cifar10":
    if args.arch == "resnet18":
        assert isinstance(net.module, Quant_ResNet) == True, type(net.module)
    elif args.arch == "resnext29_32x4d":
        assert isinstance(net.module, ResNeXt) == True, type(net.module)

# summary(net, (3,32,32))

if args.init_from and os.path.isfile(args.init_from):
    # Load checkpoint.
    print('==> Initializing from checkpoint..')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.init_from)

    net_state_dict = net.state_dict()
    net_state_dict.update(checkpoint['net'])
    net.load_state_dict(net_state_dict)

time.sleep(3)

def add_weight_decay(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                print ('add ', name, ' to no weight_decay group.')
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]

#params = add_weight_decay(net, weight_decay=args.wd, skip_keys=['.s'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)

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
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return (train_loss/batch_idx, correct/total)

    # writer.add_scalar('loss/train', train_loss/batch_idx, epoch)
    # writer.add_scalar('acc@1/train', correct/total, epoch)



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
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/ckpt.pth')
        best_acc = acc

    return (test_loss/batch_idx, correct/total)

# if args.evaluate:
print ("==> Start evaluating ...")
test(-1)
exit()

for epoch in range(start_epoch, args.epochs):
    train_loss, train_acc1 = train(epoch)
    test_loss, test_acc1 = test(epoch)
    
    if not args.lr_batch_adj:
        lr_scheduler.step()

    # log the epoch loss
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Acc1', train_acc1, epoch)
    # writer.add_scalar('Train/layer1[1].conv1.quan_w.alpha', net.module.layer1[1].conv1.quan_w.alpha, epoch)
    # writer.add_scalar('Train/layer1[1].conv1.quan_w.alpha', net.module.layer1[1].conv1.quan_w.alpha, epoch)



    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Acc1', test_acc1, epoch)
