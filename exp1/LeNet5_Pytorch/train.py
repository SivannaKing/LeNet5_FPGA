#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@AUTHOR     WZX
@EMAIL      wuzhong_xing@126.com
@TIME&LOG   2022/4/12 - download - wzx
            -----------------
            basic function
            change visdom env

            2022/4/13 - modify - wzx
            -----------------
            fix pylint error red lines(BUG002)

 TODO       google style annotation
@FUNC       plot Result data
@USAGE      >>> python train.py
            under dir [LeNet5_Pytorch]
'''


from LeNet5 import LeNet5
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from visdom import Visdom


# Hyper parameter
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10


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
data_train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, num_workers=0)


# Visdom initialization
viz = Visdom(env='LeNet5-MNIST', use_incoming_socket=False)
viz.line([0.], [0.],
         win='train_loss', opts=dict(title='train_loss'))
viz.line([[0.0, 0.0]], [0.],
         win='test', opts=dict(title='test_loss & test_acc.', legend=['loss', 'acc.']))


# Open GPU/CPU & transport
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cpu')
torch.manual_seed(1234)  # for recurrent experiment


# train
MODEL = LeNet5().to(device)
CRITERION = nn.CrossEntropyLoss().to(device)
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)


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

        loss += CRITERION(logits, y).item()  # pylint:disable=E1102
        correct += torch.eq(pred, y).sum().float().item()

    acc = correct / total

    return acc, loss


def main():
    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in range(EPOCHS):

        # Train
        for batch_idx, (data, target) in enumerate(data_train_loader):

            # data: [b, 1, 28, 28], target: [b]
            data, target = data.to(device), target.to(device)

            MODEL.train()
            logits = MODEL(data)  # pylint:disable=E1102
            loss = CRITERION(logits, target)  # pylint:disable=E1102

            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            viz.line([loss.item()], [global_step], win='train_loss', update='append')
            global_step += 1

            # train log
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(data_train_loader.dataset),
                    100. * batch_idx / len(data_train_loader), loss.item()))

        #  Test/Validation
        if epoch % 1 == 0:

            test_acc, test_loss = evaluate(MODEL, data_test_loader)
            if test_acc > best_acc:
                best_epoch = epoch+1
                best_acc = test_acc

                # save model
                torch.save(MODEL.state_dict(), 'best.pt')

                viz.line([[test_loss, test_acc]],
                         [global_step], win='test', update='append')

            # test log
            test_loss /= len(data_test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, test_acc, len(data_test_loader.dataset),
                1000000 * test_acc / len(data_test_loader.dataset)))

    print('best_epoch:', best_epoch, 'best_acc:', best_acc)

    # load best model
    MODEL.load_state_dict(torch.load('best.pt'))
    print('Successfully loaded from ckpt!\n')

    # Test
    test_acc, test_loss = evaluate(MODEL, data_test_loader)
    test_loss /= len(data_test_loader.dataset)
    print('test_acc:', test_acc, 'test_loss:', test_loss)


if __name__ == '__main__':
    main()
