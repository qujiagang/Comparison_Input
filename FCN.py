import numpy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


def relative_loss(input, target):
    loss = torch.sum(torch.abs(input-target))/torch.sum(torch.abs(target))
    return loss


data_rans_1precision = numpy.fromfile('data_rans_1precision_input_sorted')
data_rans_1precision = numpy.reshape(data_rans_1precision, (280, 320, 48, 24))
data_rans_1precision = data_rans_1precision[:, 108:211, :, :]
data_rans_1precision = numpy.transpose(data_rans_1precision, (2, 3, 0, 1)) #[48, 24, 280, 103]
data_dns = numpy.fromfile('dns_k_rans_shape')
data_dns = data_dns.reshape((280, 103, 48, 1))
data_dns = numpy.transpose(data_dns, (2, 3, 0, 1)) #[48, 1, 280, 103]

rans_data_train = data_rans_1precision[0:24, :, :, :]
dns_data_train = data_dns[0:24, :, :, :]
rans_data_validation = data_rans_1precision[25:47, :, :, :]
dns_data_validation = data_dns[25:47, :, :, :]

train_input = Variable(torch.from_numpy(rans_data_train).float())
train_output = Variable(torch.from_numpy(dns_data_train).float())
validation_input = Variable(torch.from_numpy(rans_data_validation).float())
validation_output = Variable(torch.from_numpy(dns_data_validation).float())


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(24),
            torch.nn.Conv2d(24, 24*2, 3, 1, 1),
            torch.nn.Sigmoid(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(24*2),
            torch.nn.Conv2d(24*2, 24*6, 3, 1, 1),
            torch.nn.Sigmoid(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(24*6),
            torch.nn.Conv2d(24*6, 24*2, 3, 1, 1),
            torch.nn.Sigmoid(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(24*2),
            torch.nn.Conv2d(24*2, 1, 1, 1, 0),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


cnn = CNN()
cnn_gpu = cnn.cuda()

loss_func = torch.nn.L1Loss()
# loss_func = relative_loss
optimizer = torch.optim.SGD(cnn_gpu.parameters(), lr=0.1, momentum=0.9)

train_input_gpu = train_input.cuda()
train_output_gpu = train_output.cuda()
validation_input_gpu = validation_input.cuda()
validation_output_gpu = validation_output.cuda()

t, i = 0, 0
_loss = 1.0
StartTime = []
StartTime.append(0)
loss_base = 1.0

while _loss > 0.0001:
    t = t + 1
    prediction = cnn_gpu(train_input_gpu)
    loss = loss_func(prediction, train_output_gpu)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _loss = loss.cpu().data.numpy()

    if _loss < loss_base:
        loss_base = _loss
        torch.save(cnn_gpu, 'FCN.net')

    if t % 10 == 0:
        i = i + 1
        loss_ = relative_loss(prediction, train_output_gpu)
        validation = cnn_gpu(validation_input_gpu)
        loss__ = relative_loss(validation, validation_output_gpu).cpu().data.numpy()
        StartTime.append(time.clock())
        print(t, float('%.2f' % (StartTime[i] - StartTime[i - 1])), '%f' % loss_, '%f' % loss__)


torch.save(cnn_gpu, 'FCN.net')