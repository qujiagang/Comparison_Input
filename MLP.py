import numpy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


def relative_loss(input, target):
    loss = torch.sum(torch.abs(input-target))/torch.sum(torch.abs(target))
    return loss


data_select_train = numpy.fromfile('data_select_train', dtype=float)
data_select_valition = numpy.fromfile('data_select_validation', dtype=float)

data_select_train = numpy.reshape(data_select_train, (10000, 7))
data_select_valition = numpy.reshape(data_select_valition, (5000, 7))
data_select_train = torch.from_numpy(data_select_train).float()
data_select_valition = torch.from_numpy(data_select_valition).float()


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.liner1 = torch.nn.Linear(6*1, 6*6)
        self.liner2 = torch.nn.Linear(6*6, 6*12)
        self.liner3 = torch.nn.Linear(6*12, 6*6)
        self.liner4 = torch.nn.Linear(6*6, 6*2)
        self.liner5 = torch.nn.Linear(6*2, 1)

    def forward(self, x):
        x = F.sigmoid(self.liner1(x))
        x = F.sigmoid(self.liner2(x))
        x = F.sigmoid(self.liner3(x))
        x = F.sigmoid(self.liner4(x))
        x = F.sigmoid(self.liner5(x))
        return x


net = NN()
net = net.cuda()

loss_func = torch.nn.L1Loss()
# loss_func = relative_loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

train_input = numpy.reshape(data_select_train[:, 1:7], (10000, 6))
train_output = numpy.reshape(data_select_train[:, 0], (10000, 1))
validation_input = numpy.reshape(data_select_valition[:, 1:7], (5000, 6))
validation_output = numpy.reshape(data_select_valition[:, 0], (5000, 1))
train_input = Variable(train_input)
train_output = Variable(train_output)
validation_input = Variable(validation_input)
validation_output = Variable(validation_output)

train_input_gpu = train_input.cuda()
train_output_gpu = train_output.cuda()
validation_input_gpu = validation_input.cuda()
validation_output_gpu = validation_output.cuda()

t, i = 0, 0
_loss = 1.0
StartTime = []
StartTime.append(0)
loss_base = 1.0

while _loss > 0.1:
    t = t + 1
    prediction = net(train_input_gpu)
    loss = loss_func(prediction, train_output_gpu)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _loss = loss.cpu().data.numpy()

    if _loss < loss_base:
        loss_base = _loss
        torch.save(net, 'MLP.net')

    if t % 1000 == 0:
        i = i + 1
        loss_ = relative_loss(prediction, train_output_gpu)
        validation = net(validation_input_gpu)
        loss__ = relative_loss(validation, validation_output_gpu).cpu().data.numpy()
        StartTime.append(time.clock())
        print(t, float('%.2f' % (StartTime[i] - StartTime[i - 1])), '%f' % loss_, '%f' % loss__)


torch.save(net, 'MLP.net')
