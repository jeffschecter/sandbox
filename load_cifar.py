import cPickle
import os


def LoadBatch(fname, label_kword="labels"):
    with open(fname, 'rb') as f:
        batch_data = cPickle.load(f)
    arr = batch_data["data"]
    images = arr.reshape((arr.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0
    labels = batch_data[label_kword]
    return images, labels


def Batches(include_labels=False, dataset="cifar10"):
    prefix = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        dataset)
    if dataset == "cifar10":
        batches = (
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
            "test_batch")
        label_kword = "labels"
    elif dataset == "cifar100":
        batches = ("train", "test")
        label_kword = "fine_labels"
    images, labels = [], []
    for batch in batches:
        path = os.path.join(prefix, batch)
        im, la = LoadBatch(path, label_kword=label_kword)
        images.append(im)
        labels.append(la)
    if include_labels:
        return images, labels
    else:
        return images
