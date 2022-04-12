#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@AUTHOR     WZX
@EMAIL      wuzhong_xing@126.com
@TIME&LOG   2022/4/12 - download
            -----------------
            basic function

            2022/4/12 - modify
            -----------------
            change visdom env

 TODO       google style annotation
@FUNC       plot Result data
@USAGE      >>> python train.py
            under dir [LeNet5_Pytorch]
'''

import os
from LeNet5 import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visdom import Visdom


# Hyper parameter
batch_size = 256
learning_rate = 0.001
epochs = 10


# Load Data
data_train = datasets.MNIST('./data/mnist', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Resize((28, 28)),
                                transforms.ToTensor(),  # [0,1]
                            ]))

data_test = datasets.MNIST('./data/mnist', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((28, 28)),
                               transforms.ToTensor(),  # [0,1]
                           ]))

data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=0)


# Visdom initialization
viz = Visdom(env=u'LeNet5-MNIST', use_incoming_socket=False)

viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test_loss & test_acc.',
                                                   legend=['loss', 'acc.']))


# Open GPU/CPU & transport
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cpu')
torch.manual_seed(1234)  # for recurrent experiment

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def evaluate(model, loader):

    model.eval()

    loss = 0
    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)

        loss += criterion(logits, y).item()
        correct += torch.eq(pred, y).sum().float().item()

    acc = correct / total

    return acc, loss


def main():

    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in range(epochs):

        # Train
        for batch_idx, (data, target) in enumerate(data_train_loader):

            # data: [b, 1, 28, 28], target: [b]
            data, target = data.to(device), target.to(device)

            model.train()
            logits = model(data)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='train_loss', update='append')
            global_step += 1

            # train log
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(data_train_loader.dataset),
                    100. * batch_idx / len(data_train_loader), loss.item()))

        #  Test/Validation
        if epoch % 1 == 0:

            test_acc, test_loss = evaluate(model, data_test_loader)
            if test_acc > best_acc:
                best_epoch = epoch+1
                best_acc = test_acc

                # save model
                torch.save(model.state_dict(), 'best.pt')

                viz.line([[test_loss, test_acc]],
                         [global_step], win='test', update='append')

            # test log
            test_loss /= len(data_test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, test_acc, len(data_test_loader.dataset),
                1000000 * test_acc / len(data_test_loader.dataset)))

    print('best_epoch:', best_epoch, 'best_acc:', best_acc)

    # load best model
    model.load_state_dict(torch.load('best.pt')),
    print('Successfully loaded from ckpt!\n')

    # Test
    test_acc, test_loss = evaluate(model, data_test_loader)
    test_loss /= len(data_test_loader.dataset)
    print('test_acc:', test_acc, 'test_loss:', test_loss)


if __name__ == '__main__':
    main()
