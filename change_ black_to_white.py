import six.moves.cPickle as Pickle
from PIL import Image
import numpy as np
import io

index_dir="/home/suka/PycharmProjects/pixelDTgan/"
# with open(index_dir + 'model_table.pkl', 'rb') as model:
#     model_table = Pickle.load(model)
#
#     for models_ in model_table:
#         for model_ in models_:
#             print(model_)
#             img = np.asarray(Image.open("/home/suka/eliceproject_dataset/lookbook4/data/"+model_))
#             img.setflags(write=1)
#
#             bol = np.sum(img,axis=2)<=15
#             for i in range(bol.shape[0]):
#                 for j in range(bol.shape[1]):
#                     if bol[i,j]:
#                         for channel in range(3):
#                             img[i,j,channel] = 255
#             buffer = io.BytesIO()
#             Image.fromarray(img).save(buffer, format="JPEG")
#             open("/home/suka/eliceproject_dataset/lookbook5/data/{}".format(model_), "wb").write(buffer.getvalue())

with open(index_dir + 'cloth_table.pkl', 'rb') as cloth:
    cloth_table = Pickle.load(cloth)
    for cloth_ in cloth_table:
        img = np.asarray(Image.open("/home/suka/eliceproject_dataset/lookbook4/data/"+cloth_))
        img.setflags(write=1)
        buffer = io.BytesIO()
        Image.fromarray(img).save(buffer, format="JPEG")
        open("/home/suka/eliceproject_dataset/lookbook5/data/{}".format(cloth_), "wb").write(buffer.getvalue())