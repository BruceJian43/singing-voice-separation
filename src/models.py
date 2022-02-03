import torch
import torch.nn as nn

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((5, 3), (5, 3)),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((5, 1), (5, 1)),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(), nn.ConvTranspose2d(32, 32, (5, 1), (5, 1)),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, (5, 3), (5, 3)),
            nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, x):
        embedding = self.encoder(x)
        out = self.decoder(embedding)
        return out

class DAESkipConnections(nn.Module):
    def __init__(self):
        super(DAESkipConnections, self).__init__()
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((5, 3), (5, 3))
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((5, 1), (5, 1)),
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, (5, 1), (5, 1))
        )
        self.decoder_conv4 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, (5, 3), (5, 3))
        )
        self.decoder_conv5 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1)
        )

    def forward(self, x):
        encoder_out1 = self.encoder_conv1(x)
        encoder_out2 = self.encoder_conv2(encoder_out1)
        encoder_out3 = self.encoder_conv3(encoder_out2)
        encoder_out4 = self.encoder_conv4(encoder_out3)
        decoder_out = self.decoder_conv1(encoder_out4)
        decoder_out = self.decoder_conv2(torch.cat((decoder_out, encoder_out3), dim=1))
        decoder_out = self.decoder_conv3(torch.cat((decoder_out, encoder_out2), dim=1))
        decoder_out = self.decoder_conv4(torch.cat((decoder_out, encoder_out1), dim=1))
        decoder_out = self.decoder_conv5(decoder_out)
        return decoder_out
