import cPickle
import os


def LoadBatch(file):
    with open(file, 'rb') as f:
        arr = cPickle.load(f)["data"]
    return arr.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0


def Batches():
    prefix = "./cifar10/"
    batches = (
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch")
    return [
        LoadBatch(os.path.join(prefix, batch))
        for batch in batches]
