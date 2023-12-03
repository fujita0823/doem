from source.dataset import Dataset
import os, glob, sys
import numpy as np

# calculate each class distribution
def print_class_dist(dataset):
    class_dist = [0]*9
    class_name = ["none", 'bareland','grass','developped','road','tree','water','cropland','buildings']
    for i in range(len(dataset)):
        vec = one_hot_to_label(dataset[i]["y"])
        for j in range(len(vec)):
            class_dist[j] += vec[j]
    for i in range(len(class_dist)):
        all_classes = sum(class_dist)
        print(class_name[i], class_dist[i], class_dist[i]/all_classes)

# read in the dataset
path = "../new_research/capella-oem/capella-oem"

# convert one-hot vector to class label
def one_hot_to_label(mask):
    mask = mask.numpy()
    mask = mask.reshape((mask.shape[0], mask.shape[1]*mask.shape[2]))
    label = [0]*mask.shape[0]
    for i in range(mask.shape[1]):
        label[np.argmax(mask[:,i])] += 1
    return label


cities = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
pths = []
for city in cities:
    pths_tmp = glob.glob(path+"/"+city+"/labels/*.tif")
    for pth in pths_tmp:
        pths.append(pth)

    train_dataset = Dataset(pths, classes=list(range(1,9)))
    print("city:", city)
    print_class_dist(train_dataset)