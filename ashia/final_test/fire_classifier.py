import torch
from torch.nn import BCELoss, Sigmoid
from torch import nn

class FireClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.mobileNet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).cuda()
        #self.dense1 = nn.Linear(in_features = 1000, out_features = 1, bias=True)
        self.mobileNet.classifier._modules['1'] = nn.Linear(in_features = 1280, out_features = 1, bias=True)
        self.sigmoid1 = Sigmoid().cuda()
        ##
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=64, num_layers=2)
        self.dense2 = nn.Linear(in_features = 64, out_features = 1, bias=True)
        self.sigmoid2 = Sigmoid()

    def forward(self, x, verbose=False):
        x = self.mobileNet(x)
        #x = self.dense1(x)
        x = self.sigmoid1(x)
        x = x.squeeze(1)

        return x