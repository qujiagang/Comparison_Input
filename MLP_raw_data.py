import numpy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import torch.utils.data as Data


def relative_loss(input, target):
    loss = torch.sum(torch.abs(input-target))/torch.sum(torch.abs(target))
    return loss


data_select_train = numpy.fromfile('data_select_train')
data_select_train = numpy.reshape(data_select_train, (10000, 25))
data_select_validation = numpy.fromfile('data_select_validation')
data_select_validation = numpy.reshape(data_select_validation, (5000, 25))

rans_data_train = data_select_train[:, 0:24]
dns_data_train = data_select_train[:, 24:25]
rans_data_validation = data_select_validation[:, 0:24]
dns_data_validation = data_select_validation[:, 24:25]

train_input = Variable(torch.from_numpy(rans_data_train).float())
train_output = Variable(torch.from_numpy(dns_data_train).float())
validation_input = Variable(torch.from_numpy(rans_data_validation).float())
validation_output = Variable(torch.from_numpy(dns_data_validation).float())


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(24)
        self.liner1 = torch.nn.Linear(24, 24*12)
        self.bn2 = torch.nn.BatchNorm1d(24*12)
        self.liner2 = torch.nn.Linear(24*12, 24*24)
        self.bn3 = torch.nn.BatchNorm1d(24 * 24)
        self.liner3 = torch.nn.Linear(24*24, 24*12)
        self.bn4 = torch.nn.BatchNorm1d(24 * 12)
        self.liner4 = torch.nn.Linear(24*12, 24*4)
        self.bn5 = torch.nn.BatchNorm1d(24 * 4)
        self.liner5 = torch.nn.Linear(24*4, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = torch.sigmoid(self.liner1(x))
        x = self.bn2(x)
        x = torch.sigmoid(self.liner2(x))
        x = self.bn3(x)
        x = torch.sigmoid(self.liner3(x))
        x = self.bn4(x)
        x = torch.sigmoid(self.liner4(x))
        x = self.bn5(x)
        x = torch.sigmoid(self.liner5(x))
        return x


net = NN()
net = net.cuda()

# loss_func = torch.nn.L1Loss()
loss_func = relative_loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

train_input_gpu = train_input.cuda()
train_output_gpu = train_output.cuda()
validation_input_gpu = validation_input.cuda()
validation_output_gpu = validation_output.cuda()

t, i = 0, 0
_loss = 1
StartTime = []
StartTime.append(0)

loss_ref = 1.0

while _loss > 0.0001:
    t = t + 1
    prediction = net(train_input_gpu)
    loss = loss_func(prediction, train_output_gpu)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _loss = loss.cpu().data.numpy()
    validation = net(validation_input_gpu)
    loss__ = relative_loss(validation, validation_output_gpu).cpu().data.numpy()
    if loss__<loss_ref:
        loss_ref = loss__
        torch.save(net, 'relative_loss_new/min_loss/NN-%d.net' % t)
    if t % 100 == 0:
        i = i + 1
        loss_ = relative_loss(prediction, train_output_gpu)
        validation = net(validation_input_gpu)
        loss__ = relative_loss(validation, validation_output_gpu).cpu().data.numpy()
        StartTime.append(time.clock())
        print(t, float('%.2f' % (StartTime[i] - StartTime[i - 1])), '%f' % loss_, '%f' % loss__)
        with open('relative_loss_new/loss', 'a') as file:
            file.write(str(float(loss_))+'  '+str(float(loss__))+'\n')
        torch.save(net, 'relative_loss_new/NN-%d.net' % t)


torch.save(net, 'NN2d.net')

