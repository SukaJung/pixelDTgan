import tensorflow as tf
from dataset import *
from random import shuffle
import glob
import random

shuffle_data = True  # shuffle the addresses before saving
# to shuffle data


train_filename = 'train.tfrecord'  # address to save the TFRecords file
# open the TFRecords file
data_dir="/home/suka/eliceproject_dataset/lookbook5/data/"
index_dir="/home/suka/PycharmProjects/pixelDTgan/"

data_dir = data_dir
with open(index_dir + 'cloth_table.pkl', 'rb') as cloth:
    cloth_table = Pickle.load(cloth)
with open(index_dir + 'model_table.pkl', 'rb') as model:
    model_table = Pickle.load(model)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(cloth_table)):
    # print how many images are saved every 1000 images
    # Load the image
    print(i)
    img = []
    for j in range(len(model_table[i])):
        ass_label = loadImage(data_dir+cloth_table[i])
        noass_index = random.randint(0,len(cloth_table)-1)
        noass_label = loadImage(data_dir+cloth_table[noass_index])
        img = loadImage(data_dir+model_table[i][j])
        
        # Create a feature
        feature = {'train/ass_label': _bytes_feature(tf.compat.as_bytes(ass_label.tostring())),
                   'train/img':_bytes_feature(tf.compat.as_bytes(img.tostring())),
                   'train/noass_label':_bytes_feature(tf.compat.as_bytes(noass_label.tostring()))}
   # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
        writer.write(example.SerializeToString())

writer.close()
