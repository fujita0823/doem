import os
import numpy as np
import rasterio
from torch.utils.data import Dataset as BaseDataset
from . import transforms as T
from pathlib import Path


def load_multiband(path, factor=1.0):
    src = rasterio.open(path, "r")
    out_shape = (src.count, int(src.height*factor), int(src.width*factor))
    return (np.moveaxis(src.read(out_shape=out_shape), 0, -1)).astype(np.uint8)


def load_grayscale(path, factor=1.0):
    src = rasterio.open(path, "r")
    out_shape = (src.count, int(src.height*factor), int(src.width*factor))
    return (src.read(1, out_shape=out_shape)).astype(np.uint8)

def get_crs(path):
    src = rasterio.open(path, "r")
    return src.crs, src.transform

def save_img(path,img,crs,transform):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=img.shape[1],
        width=img.shape[2],
        count=img.shape[0],
        dtype=img.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(img)
        dst.close()


class Dataset(BaseDataset):
    def __init__(self, label_list, classes=None, size=128, train=False, factor=1.0, rotate=0):
        self.fns = label_list
        if rotate == 0:
            self.augm = T.train_augm3 if train else T.valid_augm
        else:
            self.augm = T.train_augm4 if train else T.valid_augm
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale
        self.factor = factor
        city_angles = { "saitama1":130, "chiba1":-170, "tokyo1":-10, "tokyo2":-10, "tokyo3":15}
        train_fn_city = [Path(f).parents[1].name for f in label_list]
        self.angles = [rotate * city_angles[city] for city in train_fn_city]

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"), factor=self.factor)
        msk = self.load_grayscale(self.fns[idx], factor=self.factor)

        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size, self.angles[idx])
        else:
            #data = self.augm({"image": img, "mask": msk})
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)

class Dataset2(BaseDataset):
    def __init__(self, root, label_list, classes=None, size=128, train=False):
        self.fns = [os.path.join(root, "labels", x) for x in label_list]
        self.augm = T.train_augm2 if train else T.valid_augm2
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"))
        msk = self.load_grayscale(self.fns[idx])
        osm = self.load_multiband(self.fns[idx].replace("labels", "osm"))

        if self.train:
            data = self.augm({"image": img, "mask": msk, "osm": osm}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk, "osm": osm}, 512)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "z": data["osm"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)

class Dataset3(BaseDataset):
    def __init__(self, root, label_list, classes=None, size=128, train=False):
        self.fns = label_list
        self.augm = T.train_augm if train else T.valid_augm
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"))
        msk = self.load_grayscale(self.fns[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 512)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)
