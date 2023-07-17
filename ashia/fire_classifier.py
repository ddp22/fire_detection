import torch
from torch.nn import BCELoss, Sigmoid
from torch import nn

class FireClassifier(nn.Module):
    def __init__(self, batch_size = 32,
                 cuda=True,
                 out_feature_MobileNet = 256,
                 hidden_size_LSTM = 128,
                 num_layer_LSTM = 2,
                 num_frames = 1):
        super().__init__()
        self.batch_size = batch_size
        self.out_feature_MobileNet = out_feature_MobileNet
        self.hidden_size_LSTM = hidden_size_LSTM
        self.num_layer_LSTM = num_layer_LSTM
        self.num_frames = num_frames

        # MOBILENET
        self.mobileNet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.mobileNet.classifier._modules['1'] = nn.Linear(in_features = 1280,
                                                            out_features = self.out_feature_MobileNet,
                                                            bias=True)
        
        if cuda == True:
            self.mobileNet = self.mobileNet.cuda()

        # LSTM
        self.lstm = torch.nn.LSTM(input_size=self.out_feature_MobileNet,
                                  hidden_size=self.hidden_size_LSTM,
                                  num_layers=self.num_layer_LSTM,
                                  batch_first=True)
        if cuda == True:
            self.lstm = self.lstm.cuda()

        # OUTPUT LAYER
        self.fc = nn.Linear(self.hidden_size_LSTM, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x, verbose=False):
        x_size = x.size()
        real_batch_size = x_size[0]

        x = x.reshape(real_batch_size*x_size[1], x_size[2], x_size[3], x_size[4])
        x = self.mobileNet(x)
        x = x.reshape(real_batch_size, x_size[1], -1)


        h0 = torch.zeros(self.num_layer_LSTM, real_batch_size, self.hidden_size_LSTM).to(x.device)
        c0 = torch.zeros(self.num_layer_LSTM, real_batch_size,self.hidden_size_LSTM).to(x.device)
        x, _ = self.lstm(x, (h0, c0))

        x = self.fc(x[:, -1, :])
        x = self.sigmoid(x)
        x = x.squeeze(1)

        return x