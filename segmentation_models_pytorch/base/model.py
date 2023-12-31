import torch
from . import initialization as init
import math
from PIL import Image
import numpy as np
#from skimage.feature import greycomatrix
import warnings


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.decoder_dir is not None:
            init.initialize_head(self.decoder_dir)
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
        elif self.usage == "attention":
            f_l = get_laplacian(x)
            new_features = [torch.zeros_like(feature) for feature in features]
            #TODO: flag for saving figs in Attention
            attn = Attention(save_fig=False)
            for i, feature in enumerate(features):
                new_features[i] = attn(feature, f_l)
            decoder_output = self.decoder(*new_features)
        elif self.usage == "glcm":
            glcm_features = extract_glcms(x)
            new_features = [torch.zeros_like(feature) for feature in features]
            for i, feature in enumerate(features):
                print(feature.shape, glcm_features[i].shape)
                new_features[i] = feature + glcm_features[i].resize(feature.shape)
            decoder_output = self.decoder(*new_features)
        elif self.usage == "only_angle":
            input_feature = features[-1]
            if input_feature.shape[-1] == 16:
                input_feature = torch.nn.MaxPool2d(kernel_size=4, stride=4)(input_feature)
            else:
                input_feature = torch.nn.MaxPool2d(kernel_size=8, stride=8)(input_feature)
            input_feature = input_feature.view(input_feature.shape[0], -1)
            decoder_output = self.decoder_dir(input_feature)
            #for i, dec in enumerate(decoder_output):
                #print(i, dec, math.tanh(dec))
            return decoder_output
        elif self.usage == "with_angle":
            input_feature = features[-1]
            if input_feature.shape[-1] == 16:
                input_feature = torch.nn.MaxPool2d(kernel_size=4, stride=4)(input_feature)
            else:
                input_feature = torch.nn.MaxPool2d(kernel_size=8, stride=8)(input_feature)
            input_feature = input_feature.view(input_feature.shape[0], -1)
            estimated_angle = self.decoder_dir(input_feature)
            decoder_output = self.decoder(*features)
        else:
            decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if self.usage == "with_angle":
            return masks, estimated_angle

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

def get_laplacian(x):
    b,c,h,w, = x.shape
    kernel_conv = torch.nn.Conv2d(in_channels=c,out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
    # make laplacian filter
    kernel_lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).expand(1,c,3,3)
    kernel_conv.weight = torch.nn.Parameter(kernel_lap, requires_grad=False)
    kernel_conv = kernel_conv.to(x.device)
    return kernel_conv(x)

class Attention(torch.nn.Module):
    def __init__(self, save_fig=False):
        super().__init__()
        self.save_fig = save_fig
    
    def forward(self, x, f_l):
        b,c,h,w, = x.shape
        f_l = torch.nn.functional.interpolate(f_l, size=(h,w), mode='bilinear', align_corners=False).expand(b,c,h,w)

        conv_1 = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=False).to(x.device)
        conv_2 = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=False).to(x.device)

        _x = conv_1(x) + conv_2(f_l)

        if self.save_fig:
            with torch.no_grad():
                save_fig_outputs(f_l, "results/intermediate", "laplacian")
                save_fig_outputs(x, "results/intermediate", "Unet_features")
                save_fig_outputs(x+torch.sigmoid(_x), "results/intermediate", "outs")

        # sigmoid activation
        return x + torch.sigmoid(_x)

def extract_glcms(x):
    warnings.simplefilter('ignore')
    b,c,h,w, = x.shape
    glcms = []
    for i in range(b):
        input_ = (x[i,0,:,:].cpu().detach().numpy()*255).astype(np.uint8)
        glcm = greycomatrix(input_, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True)
        print(glcm.shape, "glcm shape")
        glcms.append(np.tile(glcm, (1,1,3,1)))
    return torch.tensor(glcms, dtype=torch.float32, device=x.device)

def save_fig_outputs(outputs, fout_dir, header=""):
    outputs = outputs.cpu().detach().numpy()
    for idx in range(outputs.shape[0]):
        output = outputs[idx][0]
        #output = np.transpose(output, (1,2,0))
        output = (output*255).astype(np.uint8)
        fout = fout_dir + "/" + str(idx) + f"{outputs[idx].shape[0]}_{output.shape[0]}_{header}"
        Image.fromarray(output, mode='L').save(fout+'.png')