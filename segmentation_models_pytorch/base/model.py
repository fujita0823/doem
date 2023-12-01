import torch
from . import initialization as init
import math


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x, angles = None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        if angles is not None and self.usage=="pe":
            new_features = [torch.zeros_like(feature) for feature in features]
            pe = PositionalEncoding(angles, 448, device=x.device)
            for i, feature in enumerate(features):
                new_features[i] = feature + pe(feature)
            decoder_output = self.decoder(*new_features)
        else:
            decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

    def get_features(self, x):
        return self.encoder(x)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, angles, dim, device):
        super().__init__()
        self.angles = angles
        self.dim = dim
        self.device = device
        self.pe = torch.zeros(angles.shape[0], dim, device=device)
        for i, angle in enumerate(angles):
            for j in range(0,dim,2):
                self.pe[i, j] = math.sin(angle / (10000 ** ((2*j) / dim)))
                self.pe[i, j+1] = math.cos(angle / (10000 ** ((2*(j+1)) / dim)))
        self.pe = self.pe.unsqueeze(0) 

    def forward(self, x):
        x = x*math.sqrt(self.dim)
        pe = torch.tensor(self.pe[0,:, :x.size(1)], requires_grad=False, device=self.device)
        x = (x.permute(2,3,0,1) + pe).permute(2,3,0,1)
        return x