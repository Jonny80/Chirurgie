import torch
import torch.nn as nn
import math
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import transform
from torch.autograd import Variable
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import pandas as pd


num_epochs = 5
batch_size = 100
learning_rate = 0.001


class FashionMNISTDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        data = pd.read_csv(csv_file)
        print(data.head())
        self.X = np.array(data.iloc[:,1:]).reshape(1,59999,28,28)#.astype(float)

        self.Y = np.array(data.iloc[:,0])

        del data

        self.transform = transform

    def __len__(self):
        return len(self.X)


    def __getitem__(self, idX):
        item = self.X[idX]
        label = self.Y[idX]
        if self.transform:
            item = self.transform(item)

        return (item,label)

train_dataset = FashionMNISTDataset(csv_file="/home/jonny/Dokumente/Uni/mnist/images.csv")


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};
fig = plt.figure(figsize=(8,8));
columns = 4;
rows = 5;
for i in range(1, columns*rows +1):
    print(i)
    img_xy = np.random.randint(len(train_dataset));
    img = train_dataset[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()