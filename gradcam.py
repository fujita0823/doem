import io
import os

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import timm
import unetformer
import source
import glob

# load model



class GradCam(nn.Module):
    def __init__(self, uppermodel, bottommodel):
        super().__init__()
        self.uppermodel = uppermodel
        self.bottommodel = bottommodel
        return None
    
    def infer(self, img):
        self.feature = self.uppermodel(img)  # save original feature with calcgraph 
        feat = self.feature.clone().detach().requires_grad_(True)  # -> [B,512,H/16,W/16]
        outputs = self.bottommodel(feat)
        return outputs, feat

    def forward(self, img, target, batch=0, mode="bicubic"):
        self.uppermodel.eval()
        self.bottommodel.eval()

        outputs, feat = self.infer(img)
        target = torch.argmax(target, dim=1)
        outcome = torch.argmax(outputs, dim=1)
        print(f"infer = {int(outcome)}, target = {int(target)}")

        b = batch
        B, C, H, W = feat.shape
        outputs[b][target[b]].backward(retain_graph=True)

        feat_v = feat.grad.view(B, C, H*W)  # [B, 2048, 7, 7] -> [B, 2048, 49]
        alpha = torch.mean(feat_v[b], axis=1)
        lgradcam = F.relu(torch.sum(feat[b].view(C,H,W) * alpha.view(-1,1,1), 0))
        lgradcam = F.interpolate(lgradcam.view(1,1,H,W), size=(img.shape[2], img.shape[3]), mode=mode)
        return lgradcam


def make_heatmap(img, size=(256,256), color="pink"):
    # input:
    #    img: torch tensor[C,H,W] -> [H,W]
    #    color: pyplot cmap
    # output:
    #    buf; torch tensor [3,H,W]

    buf = io.BytesIO()
    plt.figure(figsize=size, dpi=1)
    plt.gca().axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(torch.sum(img, dim=0), cmap=color)
    plt.savefig(buf, format='png')
    plt.clf()
    plt.close()
    buf = torchvision.io.decode_png(torch.frombuffer(buf.getvalue(), dtype=torch.uint8))

    return buf[0:3]  # remove alpha ch
    

root = "../new_research/capella-oem/capella-oem"
pths = glob.glob(root+"/**/labels/*.tif")
test_fns_pth = os.path.join("../new_research/capella-oem/capella-oem", "test_capella.txt")
with open(test_fns_pth) as f:
    test_fns_txt = f.read().splitlines()
test_pths = [pth for pth in pths if os.path.basename(pth) in test_fns_txt]
dataset = source.dataset.Dataset(test_pths, classes=list(range(1,9)), train=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
print(f"test_size = {len(dataset)}")

# load model
model = unetformer.UNetFormer()
model_path="results/trained/UNetFormer-swsl-resnet18_s1_UnetFormerLoss_capella_none_s1_1217_f1_r1_epoch149.pth"
model.load_state_dict(torch.load(model_path))

#gradcam = GradCam(backbone, classifier)
gradcam = GradCam(model.backbone, model.decoder)

img, target = next(iter(dataloader))
fig_image = make_heatmap(img[0], color="pink")

img_grad = gradcam(img, target, 0).detach()
fig_grad = make_heatmap(img_grad[0], color="hot")

img_grad = gradcam(img, target, 0, mode="nearest").detach()
fig_gradn = make_heatmap(img_grad[0], color="hot")

#modelseg  # segmentation model: [1,3,256,256] -> [1,1,256,256]
#msk = modelseg(img).detach()
#fig_seg = make_heatmap(msk[0], color="hot")

#fig = torch.concat([fig_image, fig_seg, fig_grad], dim=2)
#torchvision.io.write_png(fig, "vis.png")