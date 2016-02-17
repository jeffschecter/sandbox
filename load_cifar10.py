import cPickle
import gzip
import os
import shutil
import sys
import urllib2


def MaybeDownload():
    dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    if "cifar10" not in os.listdir("."):
        os.mkdir("cifar10")

    if "batches.meta" not in os.listdir("./cifar10/"):
        try:
            sys.stdout.write("Retrieving tar...")
            sys.stdout.flush()
            blob = urllib2.urlopen(dataset_url).read()
            sys.stdout.write(" fetched...")
            sys.stdout.flush()
            #TODO
        except:
            pass


def LoadBatch(file):
    with open(file, 'rb') as f:
        arr = cPickle.load(f)["data"]
    return arr.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0


def Batches():
    MaybeDownload()
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
