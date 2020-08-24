from models.resnet_quant import *
from models.resnet import *
# from trainer import Trainer
import torch
from torch import nn
from matplotlib import pyplot as plt


# net = Quant_ResNet18(bit=4)
# net = ResNet18()
# ckpt = torch.load('./checkpoint/resnet18_32_bit/epoch_119.pth')

def load_qunatize_model(bit, model_dir):
    net = Quant_ResNet18(bit=4)
    ckpt = torch.load('./checkpoint/resnet18_4_bit/epoch_119.pth')

    net_state_dict = ckpt['net']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in net_state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # net.load_state_dict(new_state_dict)

    conv1 = net.conv1
    print(conv1.quan_w(conv1.weight))

def load_original_model(bit, model_dir):
    net = ResNet18()
    ckpt = torch.load('./checkpoint/resnet18_4_bit/epoch_119.pth')

    net_state_dict = ckpt['net']

    conv1 = net.conv1
    print(conv1.quan_w(conv1.weight))


# net = Quant_ResNet18(bit=4)
# ckpt = torch.load('./checkpoint/resnet18_4_bit/epoch_119.pth')

net = ResNet18()
ckpt = torch.load('./checkpoint/resnet18_32_bit/epoch_119.pth')

net_state_dict = net.state_dict()
net_state_dict.update(ckpt['net'])

layer1_conv1 = net.layer1[0].conv1
w_1 = layer1_conv1.weight.data.numpy().reshape(-1,1)
# w_1_hat = conv1.quan_w(conv1.weight).data.numpy()


w_1_100 = w_1[:100, 0]
plt.plot(w_1_100, range(0,100), 'o')
plt.show()
#
#

# num_cols= choose the grid size you want
# def plot_kernels(tensor, num_cols=8):
#     if not tensor.ndim == 4:
#         raise Exception("assumes a 4D tensor")
#     if not tensor.shape[-1] == 3:
#         raise Exception("last dim needs to be 3 to plot")
#     num_kernels = tensor.shape[0]
#     num_rows = 1 + num_kernels // num_cols
#     fig = plt.figure(figsize=(num_cols, num_rows))
#     for i in range(tensor.shape[0]):
#         ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
#         ax1.imshow(tensor[i])
#         ax1.axis('off')
#         ax1.set_xticklabels([])
#         ax1.set_yticklabels([])
#
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#     plt.show()
#
#
# plot_kernels(filter)
