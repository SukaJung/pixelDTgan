import six.moves.cPickle as Pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from deeplab import *

def loadImage(path,ARRAY=False):
    if not ARRAY:
        inImage_ = cv2.imread(path)
        inImage = cv2.cvtColor(inImage_, cv2.COLOR_RGB2BGR)
    else:
        inImage = path
    info = np.iinfo(inImage.dtype)
    inImage = inImage.astype(np.float) / info.max

    iw = inImage.shape[1]
    ih = inImage.shape[0]
    if iw < ih:
        inImage = cv2.resize(inImage, (64, int(64 * ih/iw)))
    else:
        inImage = cv2.resize(inImage, (int(64 * iw / ih), 64))
    inImage = inImage[0:64, 0:64]
    inImage = inImage[0:64, 0:64]
    return inImage

def read_testset(MODEL):
    batch = []
    for i in range(1,7):
        image = run_segmentation(MODEL,"test/test{}.jpg".format(i))

        batch.append(loadImage(image,True))
    return batch

def scaling_img(img):
    img -= np.mean(img)
    img /= np.std(img)
    min_ = np.min(img)
    max_ = np.max(img)
    img -= min_
    img /= (max_-min_)
    return img

def plot(samples, assimg, img):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(assimg):
        ax = plt.subplot(gs[10 + i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(img):
        ax = plt.subplot(gs[20 + i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    return fig


def testplot(samples1, samples2):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples1):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(samples2):
        ax = plt.subplot(gs[len(samples1) + i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    return fig

class LookbookDataset():
    def __init__(self, data_dir, index_dir):
        self.data_dir = data_dir
        with open(index_dir+'cloth_table.pkl', 'rb') as cloth:
            self.cloth_table = Pickle.load(cloth)
        with open(index_dir+'model_table.pkl', 'rb') as model:
            self.model_table = Pickle.load(model)

        self.cn = len(self.cloth_table)
        self.path = data_dir
        self.size = 64
        self.channel = 3

    def getbatch(self, batchsize):
        ass_label = []
        noass_label = []
        img = []

        for i in range(batchsize):
#             seed = np.random.randint(1, 100000, (1,)).item()
#             np.random.seed((i+1)*seed)
            r1 = int(np.random.randint(0, self.cn, (1,)).item())
            r2 = int(np.random.randint(0, self.cn, (1,)).item())
            mn = len(self.model_table[r1])
            r3 = int(np.random.randint(0, mn, (1,)).item())

            path1 = self.cloth_table[r1]
            path2 = self.cloth_table[r2]
            path3 = self.model_table[r1][r3]
            
            img1 = loadImage(self.path + path1)
            img2 = loadImage(self.path + path2)
            img3 = loadImage(self.path + path3)
            ass_label.append(img1)
            noass_label.append(img2)
            img.append(img3)
        return ass_label, noass_label, img
    
    def getimage(self):
        return self.model_table,self.cloth_table