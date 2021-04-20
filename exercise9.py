import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import math 
''''
Tasks:
1. @DONE Check that the given implementation reaches 95% test accuracy for
   architecture input-64-64-10 in a few thousand batches.

2. @DONE, Improve initialization and check that the network learns much faster
   and reaches over 97% test accuracy.

3. @DONE Check, that with proper initialization we can train architecture
   input-64-64-64-64-64-10, while with bad initialization it does not even get off the ground.
   6 warstw, 1 wejscie ostatnia wyjscie

4. @DONE Add dropout implemented in pytorch

5. @DONE Check that with 10 hidden layers (64 units each) even with proper
    initialization the network has a hard time to start learning.

6. Implement batch normalization (use train mode also for testing
       - it should perform well enough):
    * compute batch mean and variance
    * add new variables beta and gamma
    * check that the networks learns much faster for 5 layers
    * check that the network learns even for 10 hidden layers.

Bonus task:

Design and implement in pytorch (by using pytorch functions)
   a simple convnet and achieve 99% test accuracy.

Note:
This is an exemplary exercise. MNIST dataset is very simple and we are using
it here to get resuts quickly.
To get more meaningful experience with training convnets use the CIFAR dataset.
'''


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #https://pytorch.org/docs/stable/nn.init.html
        if True:
            w = torch.empty(out_features, in_features) 
            torch.nn.init.xavier_normal_(w)
            self.weight = Parameter(w) #randn
            self.bias = Parameter(torch.randn(out_features)) #rand
        else:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.bias = Parameter(torch.Tensor(out_features))
            self.reset_parameters()

    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.5)
        init.zeros_(self.bias)

    def forward(self, x):
        r = x.matmul(self.weight.t())
        r += self.bias
        return r


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()#64-64-64-64-64-10
#        self.fc1 = Linear(784, 64)
#        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 10)
        self.dropout = nn.Dropout(0.25)
        self.linears = nn.ModuleList([
            Linear(784, 64)
        ])
        [self.linears.append(Linear(64, 64)) for x in range(10)]
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)

        x = self.fc3(x)
        return x


MB_SIZE = 128


class MnistTrainer(object):
    def __init__(self):
        transform = transforms.Compose(
                [transforms.ToTensor()])
        self.trainset = torchvision.datasets.MNIST(
            root='./data',
            download=True,
            train=True,
            transform=transform)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=MB_SIZE, shuffle=True, num_workers=4)

        self.testset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=1, shuffle=False, num_workers=4)

    def train(self):
        net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

        for epoch in range(20):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the {} test images: {} %'.format(
                total, 100 * correct / total))


def main():
    trainer = MnistTrainer()
    trainer.train()


if __name__ == '__main__':
    main()