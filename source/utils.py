import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from PIL import Image

# mpl.use("Agg")
# plt.ioff()


def progress(train_logs, valid_logs, loss_nm, metric_nm, nepochs, outdir, fn_out):
    loss_t = [dic[loss_nm] for dic in train_logs]
    loss_v = [dic[loss_nm] for dic in valid_logs]
    score_t = [dic[metric_nm] for dic in train_logs]
    score_v = [dic[metric_nm] for dic in valid_logs]

    epochs = range(0, len(score_t))
    plt.figure(figsize=(12, 5))

    # Train and validation metric
    # ---------------------------
    plt.subplot(1, 2, 1)

    idx = np.nonzero(score_t == max(score_t))[0][0]
    label = f"Train, {metric_nm}={max(score_t):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_t, "b", label=label)

    idx = np.nonzero(score_v == max(score_v))[0][0]
    label = f"Valid, {metric_nm}={max(score_v):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_v, "r", label=label)

    plt.title("Training and Validation Metric")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel(metric_nm)
    plt.ylim(0, 1)
    plt.legend()

    # Train and validation loss
    # -------------------------
    plt.subplot(1, 2, 2)
    ymax = max(max(loss_t), max(loss_v))
    ymin = min(min(loss_t), min(loss_v))
    ymax = 1 if ymax <= 1 else ymax + 0.5
    ymin = 0 if ymin <= 0.5 else ymin - 0.5

    idx = np.nonzero(loss_t == min(loss_t))[0][0]
    label = f"Train {loss_nm}={min(loss_t):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_t, "b", label=label)

    idx = np.nonzero(loss_v == min(loss_v))[0][0]
    label = f"Valid {loss_nm}={min(loss_v):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_v, "r", label=label)

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel("Loss")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(f"{outdir}/{fn_out}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    return

def label2rgb(a):
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
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in class_grey.items():
        out[a == v, 0] = class_rgb[k][0]
        out[a == v, 1] = class_rgb[k][1]
        out[a == v, 2] = class_rgb[k][2]
    return out

def save_fig_outputs(outputs, fout_dir, epoch):
    for idx in range(outputs.shape[0]):
        output = outputs[idx]
        fout = fout_dir + "/" + str(idx) + f"_epoch{str(epoch)}"
        with torch.no_grad():
            msk = torch.softmax(output, dim=0)
            msk = msk.cpu().numpy()
        y_pr = msk.argmax(axis=0).astype("uint8")
        y_pr_rgb = label2rgb(y_pr)
        Image.fromarray(y_pr_rgb).save(fout+'.png')