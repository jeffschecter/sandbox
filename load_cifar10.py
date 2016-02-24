import cPickle
import os


def LoadBatch(file):
    with open(file, 'rb') as f:
        batch_data = cPickle.load(f)
    arr = batch_data["data"]
    images = arr.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0
    labels = batch_data["labels"]
    return images, labels


def Batches(include_labels=False):
    prefix = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "cifar10/")
    batches = (
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch")
    images, labels = [], []
    for batch in batches:
        path = os.path.join(prefix, batch)
        im, la = LoadBatch(path)
        images.append(im)
        labels.append(la)
    if include_labels:
        return images, labels
    else:
        return images
