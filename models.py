import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class Generator_s2m(nn.Module):
    def __init__(self):
        super(Generator_s2m, self).__init__()
        self.conv1 = Conv_Block(3, 64, kernel_size=5)    
        self.conv2 = Conv_Block(64, 64, kernel_size=5)
        self.conv3 = Conv_Block(64, 128, kernel_size=3, stride=2)
        self.conv4 = Conv_Block(128, 128, kernel_size=3, stride=2)
        self.fc1 = Dense_Block(3200, 100)
        self.fc2 = Dense_Block(100, 100)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Classifier_s2m(nn.Module):
    def __init__(self, n_output):
        super(Classifier_s2m, self).__init__()
        self.fc = nn.Linear(100, n_output)

    def forward(self, x):
        x = self.fc(x)
        return x

class Generator_u2m(nn.Module):
    def __init__(self):
        super(Generator_u2m, self).__init__()
        self.conv1 = Conv_Block(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = Conv_Block(20, 50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.drop = nn.Dropout()
        self.fc = Dense_Block(800, 500)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x

class Classifier_u2m(nn.Module):
    def __init__(self, n_output):
        super(Classifier_u2m, self).__init__()
        self.fc = nn.Linear(500, n_output)

    def forward(self, x):
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self, task='s2m'):
        super(Net, self).__init__()
        if task == 's2m':
            self.generator = Generator_s2m()
            self.classifier = Classifier_s2m(6)
        elif task =='u2m' or task == 'm2u':
            self.generator = Generator_u2m()
            self.classifier = Classifier_u2m(6)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant = 1, adaption = False):
        x = self.generator(x)
        if adaption == True:
            x = grad_reverse(x, constant)
        x = self.classifier(x)
        return x

class disc_digits(nn.Module):
    def __init__(self,inp=4):
        super(disc_digits, self).__init__()
        self.net = nn.Sequential(nn.Linear(inp, 24),
                                 #nn.BatchNorm1d(48),
                                 nn.ReLU(),
                                 #nn.Dropout(0.2),
                                 nn.Linear(24,12),
                                 #nn.BatchNorm1d(24),
                                 nn.ReLU(),
                                 #nn.Dropout(0.2),
                                 nn.Linear(12,1))
        self.fc1 = nn.Linear(inp, 512)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        self.sgm = nn.Sigmoid()

    def forward(self, x, constant = 1, adaption = False):
        if adaption == True:
            x = grad_reverse(x, constant)
        #x = self.fc1(x)
        #x = self.relu1(x)
       # x = self.bn1(x)
        #x = self.fc2(x)
        #x = self.relu2(x)
        #x = self.bn2(x)
        #x = self.fc3(x)
        x = self.net(x)
        x = self.sgm(x)
        return x


class Net_f(nn.Module):
    def __init__(self, task='s2m'):
        super(Net_f, self).__init__()
        if task == 's2m':
            self.generator = Generator_s2m()
        elif task =='u2m' or task == 'm2u':
            self.generator = Generator_u2m()
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant = 1, adaption = False):
        x = self.generator(x)
        return x

class Net_c_cway(nn.Module):
    def __init__(self, task='s2m'):
        super(Net_c_cway, self).__init__()
        if task == 's2m':
            self.classifier = Classifier_s2m(4)
        elif task =='u2m' or task == 'm2u':
            self.classifier = Classifier_u2m(4)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant = 1, adaption = False):
        if adaption == True:
            x = grad_reverse(x, constant)
        x = self.classifier(x)
        return x

class Net_c(nn.Module):
    def __init__(self, task='s2m'):
        super(Net_c, self).__init__()
        if task == 's2m':
            self.classifier = Classifier_s2m(5)
        elif task =='u2m' or task == 'm2u':
            self.classifier = Classifier_u2m(5)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant = 1, adaption = False):
        if adaption == True:
            x = grad_reverse(x, constant)
        x = self.classifier(x)
        return x
