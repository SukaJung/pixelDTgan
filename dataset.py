import six.moves.cPickle as Pickle
import cv2
import numpy as np

def loadImage(path):
    inImage_ = cv2.imread(path)
    inImage = cv2.cvtColor(inImage_, cv2.COLOR_RGB2BGR)
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