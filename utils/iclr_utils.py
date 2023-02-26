import torch.nn as nn
import torch.nn.functional as F
# from grad_reverse import grad_reverse

def Generator(pixelda=False,nchannels=None):
    return Feature(nchannels)

def Disentangler():
    return Feature_disentangle()

def Classifier():
    return Predictor()

def Feature_Discriminator():
    return Feature_discriminator()

def Reconstructor():
    return Reconstructor()

def Mine():
    return Mine()

# %%
# class Feature(nn.Module):
#     def __init__(self,nchannels):
#         super(Feature, self).__init__()
#         self.conv1 = nn.Conv2d(nchannels, 64, kernel_size=5, stride=1, padding=2)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
#         self.bn3 = nn.BatchNorm2d(128)
#         # self.fc1 = nn.Linear(8192, 3072)
#         self.fc1 = nn.Linear(627200,3072)
#         self.bn1_fc = nn.BatchNorm1d(3072)
#         self.fc2 = nn.Linear(3072, 2048)
#         self.bn2_fc = nn.BatchNorm1d(2048)

#     def forward(self, x,reverse=False):
#         x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
#         x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
#         x = F.relu(self.bn3(self.conv3(x)))
#         # f_conv3 = x.view(x.size(0), 8192)
#         f_conv3 = x.view(x.size(0),627200)
#         f_fc1 = F.relu(self.bn1_fc(self.fc1(f_conv3)))
#         x = F.dropout(f_fc1, training=self.training)
#         # if reverse:
#         #     x = grad_reverse(x, self.lambd)
#         f_fc2 = F.relu(self.bn2_fc(self.fc2(x)))
#         return {'f_conv3':f_conv3, 'f_fc1':f_fc1, 'f_fc2':f_fc2}


# class Predictor(nn.Module):
#     def __init__(self, prob=0.5):
#         super(Predictor, self).__init__()
#         self.fc3 = nn.Linear(2048, 10)
#         self.bn_fc3 = nn.BatchNorm1d(10)
#         self.prob = prob

#     def set_lambda(self, lambd):
#         self.lambd = lambd

#     def forward(self, x, reverse=False):
#         x = self.fc3(x)
#         return x

# class Feature_discriminator(nn.Module):
#     def __init__(self, input_dim=2048):
#         super(Feature_discriminator, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, 2)
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         return x

# class Feature_disentangle(nn.Module):
#     def __init__(self):
#         super(Feature_disentangle, self).__init__()
#         self.fc1 = nn.Linear(8192, 3072)
#         self.bn1_fc = nn.BatchNorm1d(3072)
#         self.fc2 = nn.Linear(3072, 2048)
#         self.bn2_fc = nn.BatchNorm1d(2048)
#     def forward(self, x):
#         x = F.relu(self.bn1_fc(self.fc1(x)))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.bn2_fc(self.fc2(x)))
#         return x

# class Reconstructor(nn.Module):
#     def __init__(self):
#         super(Reconstructor, self).__init__()
#         self.fc = nn.Linear(4096, 8192)
#     def forward(self,x):
#         x = self.fc(x)
#         return x

# class Mine(nn.Module):
#     def __init__(self):
#         super(Mine, self).__init__()
#         self.fc1_x = nn.Linear(2048, 512)
#         self.fc1_y = nn.Linear(2048, 512)
#         self.fc2 = nn.Linear(512,1)
#     def forward(self, x,y):
#         h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
#         h2 = self.fc2(h1)
#         return h2



# %%%%%%%%

class Feature(nn.Module):
    def __init__(self,nchannels):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, 10, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(20)

        self.fc1 = nn.Linear(3920, 200)
        self.bn1_fc = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 50)
        self.bn2_fc = nn.BatchNorm1d(50)
        
        # self.fc1 = nn.Linear(320, 50)
        # self.bn1_fc = nn.BatchNorm1d(50)
        # self.fc2 = nn.Linear(50, 10)
        # self.bn2_fc = nn.BatchNorm1d(10)

    def forward(self, x,reverse=False):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        # print('a')
        x = F.relu(self.bn2(self.conv2(x)))
        # print('b')
        f_conv3 = x.view(x.size(0), 3920) #320
        # print('c')
        f_fc1 = F.relu(self.bn1_fc(self.fc1(f_conv3)))
        # print('d')
        x = F.dropout(f_fc1, training=self.training)
        # print('e')
        f_fc2 = F.relu(self.bn2_fc(self.fc2(x)))
        # print('f')
        return {'f_conv3':f_conv3, 'f_fc1':f_fc1, 'f_fc2':f_fc2}


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc3 = nn.Linear(50, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        x = self.fc3(x)
        return x

class Feature_discriminator(nn.Module):
    def __init__(self, input_dim=50):
        super(Feature_discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 2)
    def forward(self, x):
        # print('a')
        # print(x.shape)
        x = F.leaky_relu(self.fc1(x), 0.2)
        # print('b')
        # print(x.shape)
        x = F.leaky_relu(self.fc2(x), 0.2)
        # print('c')
        # print(x.shape)
        return x

class Feature_disentangle(nn.Module):
    def __init__(self):
        super(Feature_disentangle, self).__init__()
        self.fc1 = nn.Linear(3920, 200)
        self.bn1_fc = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 50)
        self.bn2_fc = nn.BatchNorm1d(50)
        
        # self.fc1 = nn.Linear(320, 50)
        # self.bn1_fc = nn.BatchNorm1d(50)
        # self.fc2 = nn.Linear(50, 10)
        # self.bn2_fc = nn.BatchNorm1d(10)
        
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x

class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.fc = nn.Linear(100, 3920)
    def forward(self,x):
        x = self.fc(x)
        return x

class Mine(nn.Module):
    def __init__(self):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(50, 10)
        self.fc1_y = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10,1)
    def forward(self, x,y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2
