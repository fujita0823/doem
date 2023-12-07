import torch
import os
from tqdm import tqdm
from .utils import save_fig_outputs
from PIL import Image
import math


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)


def train_epoch(
    model=None,
    optimizer=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
    loss_meter=None,
    score_meter=None,
    epoch=None,
    figlog_dir=None,
    use_pe=False
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()
    with tqdm(dataloader, desc="Train") as iterator:

        for idx, sample in enumerate(iterator):
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]

            optimizer.zero_grad()
            if use_pe:
                outputs = model.forward(x, sample["angle"].to(device))
            else:
                outputs = model.forward(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if figlog_dir is not None and epoch % 10 == 0: 
                train_figlog_dir = figlog_dir + "/train"
                os.makedirs(train_figlog_dir, exist_ok=True)
                save_fig_outputs(outputs, train_figlog_dir, epoch)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs


def valid_epoch(
    model=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
    epoch=None,
    figlog_dir=None,
    use_pe=False,
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    with tqdm(dataloader, desc="Valid") as iterator:
        #for sample in iterator:
        for idx, sample in enumerate(iterator):
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]

            with torch.no_grad():
                if use_pe:
                    outputs = model.forward(x, sample["angle"].to(device))
                else:
                    outputs = model.forward(x)
                loss = criterion(outputs, y)

            if figlog_dir is not None:
                valid_figlog_dir = figlog_dir + "/valid"
                os.makedirs(valid_figlog_dir, exist_ok=True)
                save_fig_outputs(outputs, figlog_dir, epoch=epoch)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)
            
            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs


def train_epoch_UNetFormer(
    model=None,
    optimizer=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
    loss_meter=None,
    score_meter=None,
    epoch=None,
    figlog_dir=None,
    use_pe=False
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()
    with tqdm(dataloader, desc="Train") as iterator:

        for idx, sample in enumerate(iterator):
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            if True:
                a = sample["angle"].to(device)
            n = x.shape[0]

            optimizer.zero_grad()
            outputs = model.forward(x)
            criterion.training = True
            loss = criterion(outputs, y, a)
            loss.backward()
            optimizer.step()

            if figlog_dir is not None and epoch % 10 == 0: 
                train_figlog_dir = figlog_dir + "/train"
                os.makedirs(train_figlog_dir, exist_ok=True)
                save_fig_outputs(outputs, train_figlog_dir, epoch)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs[0], y).cpu().detach().numpy(), n=n)

            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs


def valid_epoch_UNetFormer(
    model=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
    epoch=None,
    figlog_dir=None,
    use_pe=False,
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    with tqdm(dataloader, desc="Valid") as iterator:
        for idx, sample in enumerate(iterator):
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]

            if True:
                a = sample["angle"].to(device)

            with torch.no_grad():
                model.training = True
                outputs = model.forward(x)
                criterion.training = False
                loss = criterion(outputs, y,a)

                a = a.float() * (math.pi/180.0)
                gt_angle = torch.cat((torch.cos(a).unsqueeze(1),torch.sin(a).unsqueeze(1)), dim=1)
                angle_loss = torch.nn.MSELoss()(outputs[2], gt_angle).cpu().detach().numpy()

            if figlog_dir is not None:
                valid_figlog_dir = figlog_dir + "/valid"
                os.makedirs(valid_figlog_dir, exist_ok=True)
                save_fig_outputs(outputs, figlog_dir, epoch=epoch)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs[0], y).cpu().detach().numpy(), n=n)
            
            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            logs.update({"angle_loss": angle_loss})
            iterator.set_postfix_str(format_logs(logs))
    return logs


def train_epoch2(
    model=None,
    optimizer=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
    loss_meter=None,
    score_meter=None,
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()
    with tqdm(dataloader, desc="Train") as iterator:

        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["z"].to(device)
            n = x.shape[0]

            optimizer.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({"MSE": loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs


def valid_epoch2(
    model=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    with tqdm(dataloader, desc="Valid") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["z"].to(device)
            n = x.shape[0]

            with torch.no_grad():
                outputs = model.forward(x)
                loss = criterion(outputs, y)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)
            logs.update({"MSE": loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs



def save_fig_outputs(outputs, fout_dir, img_i):
    for idx in range(outputs.shape[0]):
        output = outputs[idx]
        fout = fout_dir + "/" + str(idx) + f"_img_{str(img_i)}"
        with torch.no_grad():
            output = output.cpu().numpy()[0]
            Image.fromarray(output, "L").save(fout+'.png')