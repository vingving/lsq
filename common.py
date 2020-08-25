import torch
import torchvision
import torchvision.transforms as transforms
import os

class data_loader():
    def __init__(self, data_name, batch_size, dataset_size, path='./data', num_workers = 8):
        self.train_loader = None
        self.test_loader = None
        if not os.path.exists(path):
            os.mkdir(path)

        if data_name == "imagenet":
            traindir = os.path.join(path, 'train/ILSVRC2012_img_train_%d'%(dataset_size))
            valdir = os.path.join(path, 'val')
            print("train dir: %s"%(traindir))
            print("valid dir: %s"% (valdir))
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

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, sampler=None)

            self.test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=512, shuffle=False,
                num_workers=num_workers, pin_memory=True)

        elif data_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            trainset = torchvision.datasets.CIFAR10(
                root=path, train=True, download=True, transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            testset = torchvision.datasets.CIFAR10(
                root=path, train=False, download=True, transform=transform_test)
            self.test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            #
            # self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
            #            'dog', 'frog', 'horse', 'ship', 'truck']

        elif data_name == "mnist":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,) )
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            trainset = torchvision.datasets.MNIST(
                root=path, train=True, download=True, transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            testset = torchvision.datasets.MNIST(
                root=path, train=False, download=True, transform=transform_test)
            self.test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # self.classes = ['0', '1', '2', '3', '4',
            #                 '5', '6', '7', '8', '9']
        elif data_name == "svhn":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            trainset = torchvision.datasets.SVHN(
                root=path, split='train', download=True, transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            testset = torchvision.datasets.SVHN(
                root=path, split='test', download=True, transform=transform_test)
            self.test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        else:
            print("Not supported dataset name")

    def get_train(self):
        return self.train_loader

    def get_test(self):
        return self.test_loader




################################################
## example code
################################################
#
# from dataLoader import data_loader
#
# loader = data_loader('cifar10', 64)
# train_loader = loader.get_train()
# valid_loader = loader.get_valid()
# test_loader = loader.get_test()

class check_point_manager():
    def __init__(self, model_name, path = './checkpoint', prefix='epoch_'):
        self.root_path = os.path.abspath(path)
        self.model_path = os.path.join(self.root_path, model_name)
        self.prefix = prefix
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def load(self, epoch = -1):
        ret = ''
        start_epoch = 0
        if epoch == -1:     #load last one
            file_list = os.listdir(self.model_path)
            s = len(self.prefix)
            if len(file_list)>0:
                last_epoch = -1
                for f in file_list:
                    tmp = int(f[s:s+3])
                    if last_epoch < tmp:
                        last_epoch = tmp
                start_epoch = last_epoch + 1
                ret = os.path.join(self.model_path, '%s%03d.pth'%(self.prefix, last_epoch))
        else:
            file = os.path.join(self.model_path, '%s%03d.pth'%(self.prefix, epoch))
            if os.path.exists(file):
                ret = file
                start_epoch = epoch + 1
            else:
                print("Can not find checkpoint [%s]", file)
        return ret, start_epoch

    def save(self, state, epoch):
        torch.save(state, os.path.join(self.model_path, '%s%03d.pth' % (self.prefix, epoch)))

#####################################################
## example code
#####################################################
# from common import data_loader, check_point_manager
#
# check_manager = check_point_manager(model_name)
# #### load ####
# save_path, start_epoch = check_manager.load()
# if start_epoch > 0:
#     # Load checkpoint.
#     print('==> Initializing from [%s]', save_path)
#     checkpoint = torch.load(save_path)
#
#     net_state_dict = net.state_dict()
#     net_state_dict.update(checkpoint)
#     net.load_state_dict(net_state_dict)
#
# #### save ####
# check_manager.save(net.state_dict(), epoch)


