import numpy as np
import torchvision.datasets as dsets
from mlxtend.data import loadlocal_mnist



path = "/home/jonny/Dokumente/Uni/mnist"

pathRaw = "/home/jonny/Dokumente/Uni/mnist/raw/"

def loadMnist(path):
    dsets.FashionMNIST(path, train=False, transform=None, target_transform=None,
                       download=True)

    X, y = loadlocal_mnist(images_path=pathRaw +"/train-images-idx3-ubyte",
                           labels_path=pathRaw + "/train-labels-idx1-ubyte")






loadMnist(path)



def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert(pathRaw + "/train-images-idx3-ubyte", pathRaw + "/train-labels-idx1-ubyte",
        path + "/mnist_train.csv", 60000)
convert(pathRaw + "/t10k-images-idx3-ubyte", pathRaw + "/t10k-labels-idx1-ubyte",
        path+"/mnist_test.csv", 10000)