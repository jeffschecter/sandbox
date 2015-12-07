import gzip
import idx2numpy
import os
import sys
import urllib2


def MaybeDownload():
    dataset_url = "http://yann.lecun.com/exdb/mnist/"
    datasets = (
        "t10k-labels-idx1-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "train-images-idx3-ubyte")

    if "mnist" not in os.listdir("."):
        os.mkdir("mnist")

    datafiles = os.listdir("./mnist/")
    for dataset in datasets:
        if dataset not in datafiles:
            try:
                sys.stdout.write("Retrieving {df}...".format(df=dataset))
                sys.stdout.flush()
                blob = urllib2.urlopen(dataset_url + dataset + ".gz").read()
                sys.stdout.write(" fetched...")
                sys.stdout.flush()
                name = "./mnist/" + dataset
                zipname = name + ".gz"
                with open(zipname, "wb") as f:
                    f.write(blob)
                sys.stdout.write(" wrote archive...")
                sys.stdout.flush()
                with gzip.open(zipname) as fin, open(name, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                sys.stdout.write(" unzipped.\n")
                sys.stdout.flush()
            except Exception as e:
                print "Failed :("
                raise e


def Datasets():
    MaybeDownload()
    prefix = "./mnist/"
    train_im = idx2numpy.convert_from_file(prefix + "train-images-idx3-ubyte")
    test_im = idx2numpy.convert_from_file(prefix + "t10k-images-idx3-ubyte")
    train_labels = idx2numpy.convert_from_file(prefix + "train-labels-idx1-ubyte")
    test_labels = idx2numpy.convert_from_file(prefix + "t10k-labels-idx1-ubyte")
    return train_im, test_im, train_labels, test_labels
