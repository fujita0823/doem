#!/usr/bin/env python
# coding: utf-8

# # OpenEarhMap Semantinc Segmentation
# 
# This demo code demonstrates training and testing of UNet-EfficientNet-B4 for the OpenEarthMap dataset (https://open-earth-map.org/). This demo code is based on the work from the "segmentation_models.pytorch" repository by qubvel, available at: https://github.com/qubvel/segmentation_models.pytorch. We extend our sincere appreciation to the original author for their invaluable contributions to the field of semantic segmentation and for providing this open-source implementation.
# 
# ---

# ### Requirements

#get_ipython().system('pip install rasterio')
#get_ipython().system('pip install pretrainedmodels')
#get_ipython().system('pip install efficientnet_pytorch')
#get_ipython().system('pip install timm')


# ### Import
# ---

# In[4]:


import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import source
import segmentation_models_pytorch as smp
import glob
import torchvision.transforms.functional as TF
import math
import cv2
from PIL import Image
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
test_mode = True

# ### Define main parameters

# In[12]:


OEM_ROOT = "./capella-oem/"
OEM_DATA_DIR = OEM_ROOT
TEST_DIR = OEM_ROOT
TRAIN_LIST = OEM_ROOT+"train_capella.txt"
VAL_LIST = OEM_ROOT+"val_capella.txt"
TEST_LIST = OEM_ROOT+"test_capella.txt"
WEIGHT_DIR = "./weight" # path to save weights
OUT_DIR = "./result" # path to save prediction images
os.makedirs(WEIGHT_DIR, exist_ok=True)
test_large = OEM_ROOT+'N35.675E139.725.tif'

seed = 0
learning_rate = 0.0001
batch_size = 4
n_epochs = 5
classes = [1, 2, 3, 4, 5, 6, 7, 8]
n_classes = len(classes)+1
classes_wt = np.ones([n_classes], dtype=np.float32)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Number of epochs   :", n_epochs)
print("Number of classes  :", n_classes)
print("Batch size         :", batch_size)
print("Device             :", device)


# ### Prepare training and validation file lists
# 
# In this demo for Google Colab, we use only two regions, i.e., Tokyo and Kyoto for training. To train with the full set, please download the OpenEarthMap dataset from https://zenodo.org/record/7223446. Note for xBD data preparation is available at https://github.com/bao18/open_earth_map.

# In[6]:


img_pths = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/labels/" in str(f)]
train_pths = [str(f) for f in img_pths if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
val_pths = [str(f) for f in img_pths if f.name in np.loadtxt(VAL_LIST, dtype=str)]
test_pths = [str(f) for f in img_pths if f.name in np.loadtxt(TEST_LIST, dtype=str)]


print("Total samples      :", len(img_pths))
print("Training samples   :", len(train_pths))
print("Validation samples :", len(val_pths))


# ### Define training and validation dataloaders

# In[7]:


trainset = source.dataset.Dataset(train_pths, classes=classes, size=512, train=True)
validset = source.dataset.Dataset(val_pths, classes=classes, train=False)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)


# ### Setup network

# In[8]:


network = smp.Unet(
    classes=n_classes,
    activation=None,
    encoder_weights="imagenet",
    encoder_name="efficientnet-b4",
    decoder_attention_type="scse",
)

# count parameters
params = 0
for p in network.parameters():
    if p.requires_grad:
        params += p.numel()

criterion = source.losses.CEWithLogitsLoss(weights=classes_wt)
criterion_name = 'CE'
metric = source.metrics.IoU2()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
network_fout = f"{network.name}_s{seed}_{criterion.name}"
OUT_DIR += network_fout # path to save prediction images
os.makedirs(OUT_DIR, exist_ok=True)

print("Model output name  :", network_fout)
print("Number of parameters: ", params)

if torch.cuda.device_count() > 1:
    print("Number of GPUs :", torch.cuda.device_count())
    network = torch.nn.DataParallel(network)
    optimizer = torch.optim.Adam(
        [dict(params=network.module.parameters(), lr=learning_rate)]
    )


# ### Visualization functions

# In[9]:


class_rgb = {
    "Bareland": [128, 0, 0],
    "Grass": [0, 255, 36],
    "Pavement": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Cropland": [75, 181, 73],
    "buildings": [222, 31, 7],
}

class_gray = {
    "Bareland": 1,
    "Grass": 2,
    "Pavement": 3,
    "Road": 4,
    "Tree": 5,
    "Water": 6,
    "Cropland": 7,
    "buildings": 8,
}

def label2rgb(a):
    """
    a: labels (HxW)
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in class_gray.items():
        out[a == v, 0] = class_rgb[k][0]
        out[a == v, 1] = class_rgb[k][1]
        out[a == v, 2] = class_rgb[k][2]
    return out


# ### Training

if not test_mode:
    # In[9]:


    start = time.time()

    max_score = 0
    train_hist = []
    valid_hist = []

    for epoch in range(n_epochs):
        print(f"\nEpoch: {epoch + 1}")

        logs_train = source.runner.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            dataloader=train_loader,
            device=device,
        )

        logs_valid = source.runner.valid_epoch(
            model=network,
            criterion=criterion,
            metric=metric,
            dataloader=valid_loader,
            device=device,
        )

        train_hist.append(logs_train)
        valid_hist.append(logs_valid)

        score = logs_valid[metric.name]

        if max_score < score:
            max_score = score
            torch.save(network.state_dict(), os.path.join(WEIGHT_DIR, f"{network_fout}.pth"))
            print("Model saved!")

    end = time.time()
    print('Processing time:',end - start)


# ### Testing
# 

# In[10]:

if test_mode:
    start = time.time()

    # load network
    network.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, f"{network_fout}_pretrained.pth")))
    network.to(device).eval()

    for fn_img in test_pths:
        img = source.dataset.load_multiband(fn_img)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        h, w = img.shape[:2]
        power = math.ceil(np.log2(h) / np.log2(2))
        shape = (2 ** power, 2 ** power)
        img = cv2.resize(img, shape)

        # test time augmentation
        imgs = []
        imgs.append(img.copy())
        imgs.append(img[:, ::-1, :].copy())
        imgs.append(img[::-1, :, :].copy())
        imgs.append(img[::-1, ::-1, :].copy())

        input = torch.cat([TF.to_tensor(x).unsqueeze(0) for x in imgs], dim=0).float().to(device)

        pred = []
        with torch.no_grad():
            msk = network(input)
            msk = torch.softmax(msk[:, :, ...], dim=1)
            msk = msk.cpu().numpy()
            pred = (msk[0, :, :, :] + msk[1, :, :, ::-1] + msk[2, :, ::-1, :] + msk[3, :, ::-1, ::-1])/4
        pred = pred.argmax(axis=0).astype("uint8")
        size = pred.shape[0:]
        y_pr = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # save image as png
        filename = os.path.splitext(os.path.basename(fn_img))[0]
        y_pr_rgb = label2rgb(y_pr)
        Image.fromarray(y_pr_rgb).save(os.path.join(OUT_DIR, filename+'_pr.png'))

    end = time.time()
    print('Processing time:',end - start)
