import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import source
import segmentation_models_pytorch as smp
#import segmentation_models_pytorch_yky as smp
import glob
import csv
import torchvision.transforms.functional as TF
import math
import cv2
from PIL import Image
from tqdm import tqdm
import time
import source.test_crop_function as tcf
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="../DEMO/capella-oem/")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--test_mode", action="store_true")
parser.add_argument("--train_city", type=str, default='all')
parser.add_argument("--img_factor", type=float, default=1.0)
parser.add_argument("--large_test",type=str,default=None)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--label_ver", type=int, default=1) # 1: first version, 2: revised version
parser.add_argument("--log_plt", action="store_true")
parser.add_argument("--log_fig", action="store_true")
parser.add_argument("--outdir", type=str, default="results")
parser.add_argument("--rotate", type=int, default=0) # 0: no rotation, 1: positive rotate, -1: negative rotate
parser.add_argument("--use_pe", action="store_true")
args = parser.parse_args()

if args.wandb:
    wandb.init(project="capella-oem")
    wandb.config.update(args)

# -----------------------
# --- Main parameters ---
# -----------------------
seed = args.seed
root = args.dataset
learning_rate = args.learning_rate
batch_size = args.batch_size
n_epochs = args.n_epochs
classes = [1, 2, 3, 4, 5, 6, 7, 8]
n_classes = len(classes)+1
classes_wt = np.ones([n_classes], dtype=np.float32)
test_mode = args.test_mode
train_city = args.train_city
large_test = args.large_test
label_ver = args.label_ver
date = "9999"

# 1: bareland
# 2: grass
# 3: pavement
# 4: road
# 5: tree
# 6: water
# 7: cropland
# 8: buildings
class_names = ['bareland','grass','developped','road','tree','water','cropland','buildings']
cities = [f for f in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, f))]

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

outdir = args.outdir
os.makedirs(outdir, exist_ok=True)
netout_dir = os.path.join(outdir, "trained")
os.makedirs(netout_dir, exist_ok=True)
csv_dir = os.path.join(outdir, "csv")
os.makedirs(csv_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Number of epochs   :", n_epochs)
print("Number of classes  :", n_classes)
print("Batch size         :", batch_size)
print("Device             :", device)

# -------------------------------------------
# --- split training and validation sets ---
# -------------------------------------------


pths = []
for city in cities:
    pths_tmp = glob.glob(root+"/"+city+"/labels/*.tif")
    for pth in pths_tmp:
        pths.append(pth)

# need change here!
train_fns = []
val_fns = []
test_fns = []
train_fns_txt = os.path.join(args.dataset,"train_capella.txt")
val_fns_txt = os.path.join(args.dataset,"val_capella.txt")
test_fns_txt = os.path.join(args.dataset,"test_capella.txt")


# read txt files
f = open(train_fns_txt, 'r')
for x in f:
    train_fns.append(x.rstrip("\n"))
f.close()
f = open(val_fns_txt, 'r')
for x in f:
    val_fns.append(x.rstrip("\n"))
f.close()
f = open(test_fns_txt, 'r')
for x in f:
    test_fns.append(x.rstrip("\n"))
f.close()

# check out if each file name is included in the list of file paths
train_pths = []
val_pths = []
test_pths = []

for pth in pths:
    fn = os.path.basename(pth)
    if fn in train_fns:
        train_pths.append(pth)
    if fn in val_fns:
        val_pths.append(pth)
    if fn in test_fns:
        test_pths.append(pth)

print("Training samples   :", len(train_pths))
print("Validation samples :", len(val_pths))
print("Test samples       :", len(test_pths))

# ---------------------------
# --- Fine tune or not ---
# ---------------------------
if train_city == 'all':
    print("Training with all cities")
else:
    print("Fine tuning with ", train_city)
    train_pths_tmp = []
    val_pths_tmp = []
    for pth_tmp in train_pths:
        if train_city in pth_tmp:
            train_pths_tmp.append(pth_tmp)
    for pth_tmp in val_pths:
        if train_city in pth_tmp:
            val_pths_tmp.append(pth_tmp)
    train_pths = train_pths_tmp
    val_pths = val_pths_tmp

# ---------------------------
# --- Define data loaders ---
# ---------------------------
trainset = source.dataset.Dataset(train_pths, classes=classes, train=True, factor=args.img_factor, rotate=args.rotate)
validset = source.dataset.Dataset(val_pths, classes=classes, train=False, factor=args.img_factor, rotate=args.rotate)
testset = source.dataset.Dataset(test_pths, classes=classes, train=False, factor=args.img_factor, rotate=args.rotate)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
if train_city != 'all':
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)
else:
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)
#test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

# --------------------------
#       network setup
# --------------------------
network = smp.Unet(
    classes=n_classes,
    activation=None,
    encoder_weights="imagenet",
    encoder_name="efficientnet-b4", #"se_resnext50_32x4d",
    decoder_attention_type="scse",
)

if args.use_pe:
    network.usage = "pe"

# count parameters
params = 0
for p in network.parameters():
    if p.requires_grad:
        params += p.numel()

criterion = source.losses.CEWithLogitsLoss(weights=classes_wt, device=device)
criterion_name = 'CE'
metric = source.metrics.IoU2()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
if label_ver == 1:
    network_fout = f"{network.name}_s{seed}_{criterion.name}_capella"
else:
    network_fout = f"{network.name}_s{seed}_{criterion.name}_capella_v2" # v3: imagenet; v4: random; v5 random & update of labels
if train_city != 'all':
    network.load_state_dict(torch.load(os.path.join(outdir, f"{network_fout}_epoch{n_epochs-1}.pth")))
    network_fout = f"{network.name}_s{seed}_{criterion.name}_{train_city}"

network_fout += "_" + str(network.usage)
network_fout += "_s"+str(seed)
network_fout += "_" + str(date)
network_fout += "_f" + str(args.img_factor)
network_fout += "_r" + str(args.rotate)

print("Model output name  :", network_fout)
print("Number of parameters: ", params)

# --------------------------
#       visualization
# --------------------------    
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

class_grey = {
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
    for k, v in class_grey.items():
        out[a == v, 0] = class_rgb[k][0]
        out[a == v, 1] = class_rgb[k][1]
        out[a == v, 2] = class_rgb[k][2]
    return out

def calc_metric(y_pr, y_gt, eps=1e-8):
    f1 = []
    iou = []
    tps = np.zeros(8)
    fns = np.zeros(8)
    fps = np.zeros(8)
    for c in np.arange(1, 9).tolist():
        if np.sum(y_gt == c) != 0:
            tp = np.logical_and(y_pr == c, y_gt == c).sum()
            fn = np.logical_and(y_pr != c, y_gt == c).sum()
            fp = np.logical_and(y_pr == c, y_gt != c).sum()
            f1.append(2 * tp / (2 * tp + fp + fn + eps))
            iou.append(tp / ( tp + fp + fn + eps))
            tps[c-1] = tp
            fns[c-1] = fn
            fps[c-1] = fp
        else:
            f1.append(np.nan)
            iou.append(np.nan)
            fp = np.logical_and(y_pr == c, y_gt != c).sum()
            fps[c-1] = fp
    return f1, iou, tps, fns, fps

if test_mode == True:
    start = time.time()
    
    figdir = os.path.join(outdir, "fig", network_fout)
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    subdir = os.path.join(outdir,"codalab",network_fout)
    os.makedirs(os.path.join(outdir, "codalabe"), exist_ok=True)
    os.makedirs(subdir, exist_ok=True)

    if os.path.exists(figdir) == False:
        os.makedirs(figdir)
    if os.path.exists(subdir) == False:
        os.makedirs(subdir)
    # header 
    header = ['city','bareland','grass','developped','road','tree','water','cropland','buildings','mIoU']
    body = []

    # load network
    pretrained_models = glob.glob(netout_dir + "/" + network_fout + "_epoch*.pth")
    if len(pretrained_models) > 0:
        max_num, name = 0, None
        for pretrained_model in pretrained_models:
            num = int(pretrained_model.split("_")[-1].split(".")[0].replace("epoch", ""))
            if num>=max_num:
                max_num, name = num, pretrained_model
        if name is not None:
            print("Loading pretrained model:", name)
            load_path = os.path.join(netout_dir, f"{network_fout}_epoch{num}.pth")
        else:
            load_path = os.path.join(netout_dir, f"{network_fout}_epoch{n_epochs-1}.pth")
            print("No pretrained model found")
    else:
        load_path = os.path.join(netout_dir, f"{network_fout}_epoch{n_epochs-1}.pth")
        print("No pretrained model found")
    #load_path = os.path.join(netout_dir, "u-efficientnet-b4_s0_CELoss_capella_s0_0000_epoch15.pth")
    network.load_state_dict(torch.load(load_path))
    network.to(device).eval()
    
    # evaluation over all folders
    
    csvfile = os.path.join(csv_dir,f"{network_fout}.csv")
    ious_all = []
    
    # tp_all, fn_all, fp_all are used to calculate IoU for all data
    tp_all = np.zeros(8)
    fn_all = np.zeros(8)
    fp_all = np.zeros(8)
        
    for k in range(len(cities)):
        city = cities[k]
        test_pths_tmp = []
        for pth_tmp in test_pths:
            if city in pth_tmp:
                test_pths_tmp.append(pth_tmp)
        # dataset
        testset = source.dataset.Dataset(test_pths_tmp, classes=classes, train=False)

        ious = []
        # tp_tmp, fn_tmp, fp_tmp are used to calculate IoU for each city
        tp_tmp = np.zeros(8)
        fn_tmp = np.zeros(8)
        fp_tmp = np.zeros(8)

        #for fn_img in test_pths_tmp:
        for fn_img in tqdm(test_pths_tmp, desc=city): 
        
            y_gt = source.dataset.load_grayscale(fn_img)
            img = source.dataset.load_multiband(str(fn_img).replace("/labels/","/images/"))

            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)  # Convert grayscale to RGB

            h, w = img.shape[:2]
            power = math.ceil(np.log2(h) / np.log2(2))
            shape = (2 ** power, 2 ** power)
            img = cv2.resize(img, shape)

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
            y_gt_rgb = label2rgb(y_gt)

            Image.fromarray(y_pr).save(os.path.join(subdir, filename+'.png'))
            Image.fromarray(y_pr_rgb).save(os.path.join(figdir, filename+'_pr.png'))
            Image.fromarray(y_gt_rgb).save(os.path.join(figdir, filename+'_gt.png'))

            f1, iou, tp, fn, fp = calc_metric(y_pr, y_gt)
            tp_tmp += tp
            fn_tmp += fn
            fp_tmp += fp
            
            tp_all += tp
            fn_all += fn
            fp_all += fp

        # change how to calculate IoU
        eps=1e-8
        ious = tp_tmp / ( tp_tmp + fp_tmp + fn_tmp + eps)
        #ious = np.nanmean(np.array(ious),0)

        data = [city]
        for c in range(n_classes-1):
            if tp_tmp[c] + fn_tmp[c] == 0: # if there is no ground truth, exclude this class from mIoU calculation
                ious[c] = np.nan
            print(class_names[c], ious[c])
            data.append(ious[c])
        print('mIoU', np.nanmean(ious)) # mIoU for each city
        data.append(np.nanmean(ious))
        body.append(data)

    # record results for all images    
    city = 'all'
    #ious = np.nanmean(np.array(ious_all),0)
    ious = tp_all / ( tp_all + fp_all + fn_all + eps)
    data = [city]
    for c in range(n_classes-1):
        print(class_names[c], ious[c])
        data.append(ious[c])
    print('mIoU', np.nanmean(ious)) # mIoU for all data
    data.append(np.nanmean(ious))
    body.append(data)

    # save in csv 
    with open(csvfile, 'w') as f: 
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)
    f.close()

    end = time.time()
    print('Processing time:',end - start)

else:
# ------------------------
# --- Network training ---
# ------------------------
    start = time.time()
    max_score = 0
    train_hist = []
    valid_hist = []
    
    if train_city != 'all':
        logs_valid = source.runner.valid_epoch(
            model=network,
            criterion=criterion,
            metric=metric,
            dataloader=valid_loader,
            device=device,
        )
        score = logs_valid[metric.name]

    if args.log_fig:
        figlog_dir = os.path.join(outdir, "figlog", network_fout)
        os.makedirs(os.path.join(outdir, "figlog"), exist_ok=True)
        os.makedirs(figlog_dir, exist_ok=True)
    else:
        figlog_dir = None    

    for epoch in range(n_epochs):
        print(f"\nEpoch: {epoch + 1}")

        logs_train = source.runner.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            figlog_dir=figlog_dir,
            use_pe=args.use_pe
        )

        logs_valid = source.runner.valid_epoch(
            model=network,
            criterion=criterion,
            metric=metric,
            dataloader=valid_loader,
            device=device,
            epoch=epoch,
            figlog_dir=figlog_dir,
            use_pe=args.use_pe
        )

        train_hist.append(logs_train)
        valid_hist.append(logs_valid)

        if args.wandb:
            wb_logs = {}
            wb_logs["train_loss"] = logs_train[criterion.name]
            wb_logs["train_iou"] = logs_train[metric.name]
            wb_logs["valid_loss"] = logs_valid[criterion.name]
            wb_logs["valid_iou"] = logs_valid[metric.name]
            wandb.log(wb_logs)
       
        score = logs_valid[metric.name]

        if args.log_plt:
            prog_dir = os.path.join(outdir, "progress")
            source.utils.progress(
                train_hist,
                valid_hist,
                criterion.name,
                metric.name,
                n_epochs,
                outdir,
                network_fout,
            )
        
        if max_score < score or epoch == n_epochs-1:
            max_score = score
            #torch.save(network, os.path.join(outdir, f"{network_fout}.pth"))
            torch.save(network.state_dict(), os.path.join(netout_dir, f"{network_fout}_epoch{epoch}.pth"))
            print("Model saved!")

        if args.log_plt and epoch % 20 == 0 and epoch != 0:
           learning_rate *= 1e-1
           optimizer.param_groups[0]["lr"] = learning_rate
           print("Decrease decoder learning rate to {}".format(learning_rate))

    print(f"Completed: {(time.time() - start)/60.0:.4f} min.")

if args.wandb:
    wandb.finish()
